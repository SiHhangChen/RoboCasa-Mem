"""WR07 adapter for the Omega-conditioned policy in policy/baseline."""

from __future__ import annotations

import dataclasses
import hashlib
import math
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from bench.evaluation.action import validate_action_chunk
from bench.policy.backends.openpi import OpenPIBackend
from bench.policy.checkpoint import resolve_checkpoint_identity
from bench.policy.protocol import InferenceRequest, InferenceResponse
from bench.real_robot.contracts import ACTION_DIM, CAMERA_ALIASES, RUNTIME_PROFILE, STATE_DIM
from bench.real_robot.wr07_prompts import FULL_CONFIG, SUBGOAL_PROMPTS

CAMERA_MAP = {
    "agentview_left": "front",
    "agentview_right": "left",
    "eye_in_hand": "right",
}


def _required_path(path: Path, *, directory: bool) -> Path:
    resolved = path.expanduser().resolve()
    if not (resolved.is_dir() if directory else resolved.is_file()):
        kind = "directory" if directory else "file"
        raise FileNotFoundError(f"Required {kind} does not exist: {resolved}")
    return resolved


def _fingerprint_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


class CurrentOmegaEncoder:
    """Online equivalent of the six-step current-slot Omega cache builder."""

    def __init__(self, repo: Path, checkpoint: Path, *, device: str = "cuda:0") -> None:
        import torch

        self.device = device
        repo = _required_path(repo, directory=True)
        checkpoint = _required_path(checkpoint, directory=False)
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        from vggt_omega.models import VGGTOmega

        model = VGGTOmega(enable_camera=False, enable_depth=False, enable_alignment=False)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        for key in ("state_dict", "model"):
            if isinstance(state, dict) and isinstance(state.get(key), dict):
                state = state[key]
        if not isinstance(state, dict):
            raise TypeError(f"Expected Omega state dict in {checkpoint}")
        if state and all(str(key).startswith("module.") for key in state):
            state = {str(key).removeprefix("module."): value for key, value in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if any(key.startswith("aggregator.") for key in missing) or any(
            not key.startswith(("camera_head.", "dense_head.", "text_alignment_head."))
            for key in unexpected
        ):
            raise RuntimeError(
                f"Incompatible Omega checkpoint: missing={missing[:8]}, unexpected={unexpected[:8]}"
            )
        self.model = model.to(device).eval()
        self.reset()

    def reset(self) -> None:
        self._frames: deque[tuple[Any, ...]] = deque(maxlen=6)

    @staticmethod
    def preprocess(image: np.ndarray) -> Any:
        import torch
        from PIL import Image

        array = np.asarray(image)
        if array.ndim != 3 or array.shape[-1] != 3 or array.dtype != np.uint8:
            raise ValueError(f"Omega expects raw HWC RGB uint8, got {array.dtype}{array.shape}")
        picture = Image.fromarray(array).convert("RGB")
        width, height = picture.size
        aspect = height / width
        if aspect < 0.5:
            crop_width = min(width, max(1, round(height / 0.5)))
            left = max((width - crop_width) // 2, 0)
            picture = picture.crop((left, 0, left + crop_width, height))
        elif aspect > 2.0:
            crop_height = min(height, max(1, round(width * 2.0)))
            top = max((height - crop_height) // 2, 0)
            picture = picture.crop((0, top, width, top + crop_height))
        width, height = picture.size
        aspect = height / width
        patches = (256 // 16) ** 2
        width_patches = max(1, round(math.sqrt(patches / aspect)))
        height_patches = max(1, round(patches / width_patches))
        picture = picture.resize(
            (width_patches * 16, height_patches * 16), Image.Resampling.BICUBIC
        )
        return torch.from_numpy(np.asarray(picture, dtype=np.float32) / 255.0).permute(2, 0, 1)

    def encode(self, images: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        import torch
        from torch.nn import functional

        if set(images) != set(CAMERA_ALIASES):
            raise ValueError(f"Omega requires synchronized cameras {CAMERA_ALIASES}")
        self._frames.append(tuple(self.preprocess(images[alias]) for alias in CAMERA_ALIASES))
        history = list(reversed(self._frames))
        history.extend([history[-1]] * (6 - len(history)))
        views = [view for frame in history for view in frame]
        height = max(view.shape[-2] for view in views)
        width = max(view.shape[-1] for view in views)
        padded = [
            functional.pad(view, (0, width - view.shape[-1], 0, height - view.shape[-2]), value=1.0)
            for view in views
        ]
        batch = torch.stack(padded)[None].to(self.device)
        with torch.inference_mode():
            tokens = self.model(batch)["camera_and_register_tokens"][:, :3]
        if tuple(tokens.shape) != (1, 3, 17, 2048):
            raise ValueError(f"Unexpected Omega output shape {tuple(tokens.shape)}")
        memory = tokens[0].float().cpu().numpy().astype(np.float16).astype(np.float32)
        return memory[None, :, None, :, :], np.ones((1,), dtype=np.bool_)


class BaselineWR07Backend(OpenPIBackend):
    def __init__(
        self,
        policy: Any,
        encoder: Any,
        *,
        checkpoint: Any,
        omega_fingerprint: str,
        stats_fingerprint: str,
        code_revision: dict[str, str] | None = None,
    ) -> None:
        super().__init__(
            policy,
            config_name=FULL_CONFIG,
            checkpoint=checkpoint,
            action_horizon=50,
            image_size=(224, 224),
            camera_map=CAMERA_MAP,
            runtime_profile=RUNTIME_PROFILE,
            prompt_contract={
                "prompt_mode": "episode_choice",
                "prompt_choices": dict(SUBGOAL_PROMPTS),
            },
            server_id="baseline-wr07-omega-current",
            code_revision=code_revision,
        )
        self.encoder = encoder
        self._contract = dataclasses.replace(
            self.contract,
            observation=dataclasses.replace(
                self.contract.observation,
                required_modalities=("rgb", "state", "prompt", "raw_images"),
            ),
            policy={
                **self.contract.policy,
                "family": "baseline_wr07_omega",
                "omega_input_shape": [1, 3, 1, 17, 2048],
                "omega_checkpoint_fingerprint": omega_fingerprint,
                "norm_stats_fingerprint": stats_fingerprint,
                "decision_stride_frames": 10,
                "omega_temporal_steps": 6,
            },
        )
        self._last_step: int | None = None

    def reset(self, episode_id: str) -> None:
        super().reset(episode_id)
        self.encoder.reset()
        self._last_step = None

    @classmethod
    def from_checkpoint(
        cls,
        *,
        checkpoint_dir: Path,
        dataset_root: Path,
        omega_repo: Path,
        omega_checkpoint: Path,
        omega_device: str = "cuda:0",
        code_revision: dict[str, str] | None = None,
    ) -> BaselineWR07Backend:
        from openpi.policies import policy_config
        from openpi.training import config as training_config

        checkpoint = _required_path(checkpoint_dir, directory=True)
        if (
            not (checkpoint / "params").is_dir()
            and not (checkpoint / "model.safetensors").is_file()
        ):
            raise FileNotFoundError(f"No policy weights under {checkpoint}")
        dataset = _required_path(dataset_root, directory=True)
        stats_path = _required_path(dataset / "meta" / "stats.json", directory=False)
        omega_file = _required_path(omega_checkpoint, directory=False)
        train_config = training_config.get_config(FULL_CONFIG)
        if (
            not train_config.model.omega_enabled
            or train_config.model.omega_memory_horizon != 1
            or train_config.model.omega_num_layers != 1
            or train_config.model.action_horizon != 50
            or train_config.data.state_keep_dim != STATE_DIM
            or train_config.data.action_output_dim != ACTION_DIM
            or train_config.data.omega_memory_mode != "current"
            or train_config.data.omega_stride != 10
            or not train_config.data.prompt_from_subtask
            or dict(train_config.data.camera_mapping)
            != {key: f"observation.images.{alias}" for key, alias in CAMERA_MAP.items()}
        ):
            raise ValueError(
                f"{FULL_CONFIG} differs from the expected WR07 Omega training contract"
            )
        data = dataclasses.replace(train_config.data, dataset_path=str(dataset))
        train_config = dataclasses.replace(train_config, data=data)
        data_config = data.create(train_config.assets_dirs, train_config.model)
        stats = data_config.norm_stats
        if (
            stats is None
            or np.asarray(stats["state"].mean).shape != (STATE_DIM,)
            or np.asarray(stats["actions"].mean).shape != (ACTION_DIM,)
        ):
            raise ValueError("WR07 normalization statistics must contain state17 and action17")
        policy = policy_config.create_trained_policy(
            train_config,
            checkpoint,
            norm_stats=stats,
        )
        encoder = CurrentOmegaEncoder(omega_repo, omega_file, device=omega_device)
        return cls(
            policy,
            encoder,
            checkpoint=resolve_checkpoint_identity(checkpoint),
            omega_fingerprint=_fingerprint_file(omega_file),
            stats_fingerprint=_fingerprint_file(stats_path),
            code_revision=code_revision,
        )

    def infer(self, request: InferenceRequest) -> InferenceResponse:
        observation = request.observation
        if observation.prompt not in SUBGOAL_PROMPTS.values():
            raise ValueError("WR07 baseline prompt must be one of the three trained colors")
        if self._active_prompt is not None and observation.prompt != self._active_prompt:
            raise ValueError("WR07 baseline prompt cannot change within an episode")
        expected_step = 0 if self._last_step is None else self._last_step + 10
        if request.step != expected_step:
            raise ValueError(
                f"WR07 baseline expected decision frame {expected_step}, got {request.step}"
            )
        raw = observation.extra_modalities.get("raw_images")
        if not isinstance(raw, dict) or set(raw) != set(CAMERA_ALIASES):
            raise ValueError("WR07 baseline requires raw_images for all three cameras")
        raw_images = {alias: np.asarray(raw[alias]) for alias in CAMERA_ALIASES}
        if any(
            image.dtype != np.uint8 or image.shape != (480, 640, 3) for image in raw_images.values()
        ):
            raise ValueError("raw_images must contain 480x640 HWC RGB uint8 images")
        self._active_prompt = observation.prompt
        start = time.monotonic()
        memory, mask = self.encoder.encode(raw_images)
        omega_ms = (time.monotonic() - start) * 1000.0
        if memory.shape != (1, 3, 1, 17, 2048) or mask.shape != (1,):
            raise ValueError("Online Omega encoder returned an incompatible feature shape")
        policy_input = {
            "images": {key: raw_images[value] for key, value in CAMERA_MAP.items()},
            "state": observation.state,
            "prompt": observation.prompt,
            "omega_memory": memory,
            "omega_memory_mask": mask,
        }
        output = self.policy.infer(policy_input)
        actions = validate_action_chunk(
            np.asarray(output["actions"]),
            expected_horizon=self.contract.action.horizon,
            expected_dim=self.contract.action.dim,
            action_schema=self.contract.action.schema,
        )
        self._last_step = request.step
        return InferenceResponse(
            request_id=request.request_id,
            episode_id=request.episode_id,
            actions=actions,
            timing={
                "omega_ms": omega_ms,
                "model_ms": (time.monotonic() - start) * 1000.0 - omega_ms,
            },
        )
