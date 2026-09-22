"""GR00T N1.7 backend for the membench policy protocol.

Loads a LoRA checkpoint trained by the NVIDIA Isaac-GR00T stack through the
LeRobot 0.6.1 integration (installed package or local source checkout):
the checkpoint directory carries a PEFT adapter plus serialized pre/post
processors, and the adapter keys are prefixed with `_groot_model.` because the
adapter was trained on a `GrootPolicy` wrapper. Loading therefore goes through
`GrootPolicy` + `PeftModel`, never the bare `GR00TN17` model.
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import torch

from bench.evaluation.action import validate_action_chunk
from bench.policy.checkpoint import resolve_checkpoint_identity
from bench.policy.protocol import (
    ActionContract,
    CheckpointIdentity,
    InferenceRequest,
    InferenceResponse,
    ObservationContract,
    PolicyContract,
)
from bench.runtime.data_contract import ACTION_DIM, ACTION_SCHEMA, STATE_DIM, STATE_SCHEMA

# Wire camera aliases -> dataset camera keys consumed by the checkpoint's
# processor pipeline (rename_map is empty in the serialized preprocessor, so
# inference must present the dataset's original column names).
CAMERA_KEY_MAP: dict[str, str] = {
    "agentview_left": "robot0_agentview_left",
    "agentview_right": "robot0_agentview_right",
    "eye_in_hand": "robot0_eye_in_hand",
}
REQUIRED_CAMERAS: tuple[str, ...] = tuple(CAMERA_KEY_MAP)
LOGGER = logging.getLogger(__name__)


class Gr00TBackend:
    def __init__(
        self,
        policy: Any,
        *,
        preprocessor: Any,
        postprocessor: Any,
        checkpoint: CheckpointIdentity,
        config_name: str,
        action_horizon: int,
        image_size: tuple[int, int],
        state_schema: str = STATE_SCHEMA,
        server_id: str | None = None,
        code_revision: dict[str, str] | None = None,
        seed_by_episode: bool = False,
    ) -> None:
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.config_name = config_name
        self.seed_by_episode = bool(seed_by_episode)
        self._contract = PolicyContract(
            server_id=server_id or f"gr00t-{config_name}",
            policy={
                "family": "gr00t",
                "name": config_name,
                "config": config_name,
                "seed_policy": "episode_seed_v1" if seed_by_episode else "unseeded",
            },
            checkpoint=checkpoint,
            observation=ObservationContract(
                state_schema=state_schema,
                state_dim=STATE_DIM,
                required_cameras=REQUIRED_CAMERAS,
                image_size=image_size,
            ),
            action=ActionContract(
                schema=ACTION_SCHEMA,
                dim=ACTION_DIM,
                horizon=action_horizon,
            ),
            control_frequency_hz=20.0,
            code_revision=dict(code_revision or {}),
        )

    @classmethod
    def from_checkpoint(
        cls,
        *,
        checkpoint_dir: Path,
        base_model_dir: Path,
        vlm_model_dir: Path | None = None,
        image_size: tuple[int, int],
        state_schema: str,
        pytorch_device: str | None = None,
        server_id: str | None = None,
        code_revision: dict[str, str] | None = None,
        seed_by_episode: bool = False,
    ) -> Gr00TBackend:
        try:
            from lerobot.configs import FeatureType, PolicyFeature
            from lerobot.policies.groot.configuration_groot import GrootConfig
            from lerobot.policies.groot.modeling_groot import GrootPolicy
            from lerobot.policies.groot.processor_groot import (
                make_groot_pre_post_processors_from_pretrained,
            )
        except ImportError as exc:
            raise RuntimeError(
                "LeRobot with GR00T N1.7 support is not importable. Run "
                "serve_gr00t.py in the LeRobot 0.6.1 serving environment."
            ) from exc

        ckpt = checkpoint_dir.expanduser().resolve()
        if not (ckpt / "train_config.json").is_file():
            raise FileNotFoundError(f"GR00T checkpoint must contain train_config.json: {ckpt}")
        train_config = json.loads((ckpt / "train_config.json").read_text(encoding="utf-8"))
        policy_section = train_config["policy"]
        base_model = base_model_dir.expanduser().resolve()

        features: dict[str, PolicyFeature] = {}
        for key, spec in policy_section["input_features"].items():
            ftype = {
                "VISUAL": FeatureType.VISUAL,
                "STATE": FeatureType.STATE,
                "ACTION": FeatureType.ACTION,
            }[spec["type"]]
            features[key] = PolicyFeature(type=ftype, shape=tuple(spec["shape"]))
        output_features = {
            key: PolicyFeature(type=FeatureType.ACTION, shape=tuple(spec["shape"]))
            for key, spec in policy_section["output_features"].items()
        }
        features.update(output_features)

        config_kwargs = {
            "base_model_path": str(base_model),
            "embodiment_tag": policy_section.get("embodiment_tag", "robocasa_panda_omron"),
            "chunk_size": int(policy_section.get("chunk_size", 40)),
            "n_action_steps": int(policy_section.get("n_action_steps", 40)),
            "max_state_dim": int(policy_section.get("max_state_dim", 132)),
            "max_action_dim": int(policy_section.get("max_action_dim", 132)),
            "model_params_fp32": False,  # inference: keep bf16 weights
            "use_flash_attention": False,  # SDPA fallback
            "input_features": features,
            "output_features": output_features,
        }
        # Some Isaac-GR00T forks expose this override, while released LeRobot
        # 0.6.1 reads the backbone path from the base model's own config.json.
        # The tokenizer override below is independent and works on both.
        supported_fields = {item.name for item in fields(GrootConfig)}
        if "backbone_model_path" in supported_fields:
            config_kwargs["backbone_model_path"] = (
                None if vlm_model_dir is None else str(vlm_model_dir.expanduser().resolve())
            )
        for name in (
            "n_obs_steps",
            "use_bf16",
            "use_relative_actions",
            "relative_exclude_joints",
            "num_inference_timesteps",
            "rtc_ramp_rate",
            "action_decode_transform",
        ):
            if name in supported_fields and name in policy_section:
                config_kwargs[name] = policy_section[name]
        config = GrootConfig(**config_kwargs)

        policy = GrootPolicy(config)
        from peft import PeftModel

        policy = PeftModel.from_pretrained(policy, ckpt)
        device = pytorch_device or ("cuda" if torch.cuda.is_available() else "cpu")
        policy = policy.to(device, dtype=torch.bfloat16)
        policy.eval()

        # The serialized vlm_encode step points at the training machine's
        # Cosmos-Reason2-2B path; override it with the local copy when the
        # recorded path does not exist (or when the caller says so).
        preprocessor_overrides: dict[str, Any] | None = None
        serialized_path = _serialized_vlm_model_dir(ckpt)
        if vlm_model_dir is not None:
            preprocessor_overrides = {
                "groot_n1_7_vlm_encode_v1": {
                    "model_name": str(vlm_model_dir.expanduser().resolve())
                }
            }
        elif serialized_path is not None and not serialized_path.exists():
            raise FileNotFoundError(
                "vlm_encode tokenizer path recorded in the checkpoint does not "
                f"exist here: {serialized_path}. Pass --vlm-model-dir pointing at "
                "a local Cosmos-Reason2-2B copy."
            )
        preprocessor, postprocessor = make_groot_pre_post_processors_from_pretrained(
            config,
            str(ckpt),
            preprocessor_overrides=preprocessor_overrides,
        )

        config_name = train_config.get("job_name") or ckpt.parent.name
        return cls(
            policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            checkpoint=resolve_checkpoint_identity(ckpt, step=_checkpoint_step(ckpt)),
            config_name=config_name,
            action_horizon=int(policy_section.get("chunk_size", 40)),
            image_size=image_size,
            state_schema=state_schema,
            server_id=server_id,
            code_revision=code_revision,
            seed_by_episode=seed_by_episode,
        )

    @property
    def contract(self) -> PolicyContract:
        return self._contract

    def reset(self, episode_id: str) -> None:
        if self.seed_by_episode:
            seed = _episode_seed(episode_id)
            random.seed(seed)
            np.random.seed(seed % (2**32))
            # torch.manual_seed also seeds the CUDA generators. Do this for
            # every episode, not just process startup, so resume is independent
            # of the number of previous requests or warmup calls.
            torch.manual_seed(seed)
            LOGGER.info("Reset episode %s with model seed %d", episode_id, seed)
        reset = getattr(self.policy, "reset", None)
        if callable(reset):
            reset()

    @torch.inference_mode()
    def infer(self, request: InferenceRequest) -> InferenceResponse:
        observation = request.observation
        batch: dict[str, Any] = {}
        for alias, dataset_key in CAMERA_KEY_MAP.items():
            image = np.asarray(observation.images[alias])
            # Wire images are uint8 HWC; the processor expects (B, C, H, W) tensors.
            image = np.ascontiguousarray(np.transpose(image, (2, 0, 1)))
            batch[f"observation.images.{dataset_key}"] = torch.from_numpy(image)[None]
        batch["observation.state"] = torch.from_numpy(
            np.ascontiguousarray(observation.state, dtype=np.float32)
        )[None]
        batch["task"] = [observation.prompt]

        start = time.monotonic()
        collated = self.preprocessor(batch)
        chunk = self.policy.predict_action_chunk(collated)
        decoded = self.postprocessor(chunk)
        elapsed_ms = (time.monotonic() - start) * 1000.0

        if torch.is_tensor(decoded):
            # NumPy has no native bfloat16 dtype. Cast before crossing the
            # torch/NumPy boundary, including when postprocessing kept bf16.
            actions = decoded.detach().to(device="cpu", dtype=torch.float32).numpy()
        else:
            actions = np.asarray(decoded)
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        actions = validate_action_chunk(
            actions,
            expected_horizon=self.contract.action.horizon,
            expected_dim=self.contract.action.dim,
            action_schema=self.contract.action.schema,
        )
        return InferenceResponse(
            request_id=request.request_id,
            episode_id=request.episode_id,
            actions=actions,
            timing={"model_ms": elapsed_ms},
        )

    def close(self) -> None:
        close = getattr(self.policy, "close", None)
        if callable(close):
            close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _episode_seed(episode_id: str) -> int:
    match = re.search(r"(?:^|-)seed-(\d+)(?:-|$)", episode_id)
    if match is None:
        raise ValueError(
            f"seed_by_episode requires an episode ID containing 'seed-<integer>': {episode_id!r}"
        )
    return int(match.group(1))


def _checkpoint_step(checkpoint_dir: Path) -> int | None:
    state_path = checkpoint_dir.parent / "training_state" / "training_step.json"
    if state_path.is_file():
        step = json.loads(state_path.read_text(encoding="utf-8"))["step"]
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError(f"Invalid checkpoint training step in {state_path}")
        return step
    name = checkpoint_dir.parent.name if checkpoint_dir.name == "pretrained_model" else ""
    return int(name) if name.isdecimal() else None


def _serialized_vlm_model_dir(checkpoint_dir: Path) -> Path | None:
    path = checkpoint_dir / "policy_preprocessor.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    for step in payload.get("steps", ()):
        if step.get("registry_name") == "groot_n1_7_vlm_encode_v1":
            model_name = step.get("config", {}).get("model_name")
            if isinstance(model_name, str):
                return Path(model_name)
    return None
