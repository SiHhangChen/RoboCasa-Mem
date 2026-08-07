"""LeRobot ACT backend with strict MobileMemBench v2 validation."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from bench.evaluation.action import validate_action_chunk
from bench.policy.protocol import (
    ActionContract,
    InferenceRequest,
    InferenceResponse,
    ObservationContract,
    PolicyContract,
)
from bench.policy.checkpoint import (
    read_checkpoint_metadata,
    resolve_checkpoint_identity,
)
from bench.runtime.data_contract import ACTION_DIM, ACTION_SCHEMA, STATE_DIM, STATE_SCHEMA


STATE_FEATURE = "observation.state"
ACTION_FEATURE = "action"
CAMERA_PREFIX = "observation.images."


class LeRobotACTBackend:
    def __init__(
        self,
        *,
        policy: Any,
        preprocessor: Any,
        postprocessor: Any,
        contract: PolicyContract,
        device: str,
        torch_module: Any,
    ) -> None:
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self._contract = contract
        self.device = device
        self.torch = torch_module
        self.camera_features = dict(contract.policy.get("camera_features", {}))

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: Path,
        *,
        device: str = "cuda",
        checkpoint_uri: str | None = None,
        server_id: str | None = None,
        code_revision: dict[str, str] | None = None,
    ) -> "LeRobotACTBackend":
        checkpoint_path = resolve_pretrained_model_dir(checkpoint_dir)
        inspected = inspect_lerobot_checkpoint(checkpoint_path)
        try:
            import torch
            from lerobot.configs import PreTrainedConfig
            from lerobot.policies.act.modeling_act import ACTPolicy
            from lerobot.policies.factory import make_pre_post_processors
        except ImportError as exc:
            raise RuntimeError(
                "LeRobot is not importable. Run serve_lerobot.py from this repository "
                "or install policy/lerobot."
            ) from exc

        policy_config = PreTrainedConfig.from_pretrained(checkpoint_path)
        if str(policy_config.type) != "act":
            raise ValueError(
                f"serve_lerobot currently requires an ACT checkpoint, got {policy_config.type!r}"
            )
        policy_config.device = device
        policy = ACTPolicy.from_pretrained(
            checkpoint_path,
            config=policy_config,
        ).to(device)
        policy.eval()
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy_config,
            pretrained_path=str(checkpoint_path),
            preprocessor_overrides={"device_processor": {"device": device}},
            postprocessor_overrides={"device_processor": {"device": "cpu"}},
        )
        contract = PolicyContract(
            server_id=server_id or f"lerobot-act-{checkpoint_path.parent.name}",
            policy={
                "family": "lerobot",
                "name": "act",
                "config": "act",
                "camera_features": inspected["camera_features"],
            },
            checkpoint=resolve_checkpoint_identity(
                checkpoint_path,
                uri=checkpoint_uri,
            ),
            observation=ObservationContract(
                state_schema=inspected["state_schema"],
                state_dim=STATE_DIM,
                required_cameras=("agentview_left", "agentview_right", "eye_in_hand"),
                image_size=tuple(inspected["image_size"]),
            ),
            action=ActionContract(
                schema=ACTION_SCHEMA,
                dim=ACTION_DIM,
                horizon=int(inspected["horizon"]),
            ),
            control_frequency_hz=20.0,
            code_revision=dict(code_revision or {}),
        )
        return cls(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            contract=contract,
            device=device,
            torch_module=torch,
        )

    @property
    def contract(self) -> PolicyContract:
        return self._contract

    def validate_checkpoint_contract(self) -> None:
        if self.contract.observation.state_dim != STATE_DIM:
            raise ValueError(f"LeRobot ACT checkpoint must use state_dim={STATE_DIM}")

    def reset(self, episode_id: str) -> None:
        del episode_id
        self.policy.reset()

    def infer(self, request: InferenceRequest) -> InferenceResponse:
        batch: dict[str, Any] = {
            STATE_FEATURE: self.torch.from_numpy(request.observation.state),
        }
        for alias in self.contract.observation.required_cameras:
            image = np.asarray(request.observation.images[alias], dtype=np.float32) / 255.0
            batch[self.camera_features[alias]] = self.torch.from_numpy(
                np.ascontiguousarray(image.transpose(2, 0, 1))
            )
        start = time.monotonic()
        with self.torch.inference_mode():
            processed = self.preprocessor(batch)
            normalized_actions = self.policy.predict_action_chunk(processed)
            actions = self.postprocessor(normalized_actions)
        model_ms = (time.monotonic() - start) * 1000.0
        if hasattr(actions, "detach"):
            actions = actions.detach().cpu().numpy()
        actions = np.asarray(actions)
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        actions = validate_action_chunk(
            actions,
            expected_horizon=self.contract.action.horizon,
        )
        return InferenceResponse(
            request_id=request.request_id,
            episode_id=request.episode_id,
            actions=actions,
            timing={"model_ms": model_ms},
        )

    def close(self) -> None:
        self.policy = None
        if self.device.startswith("cuda") and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def resolve_pretrained_model_dir(checkpoint_dir: Path) -> Path:
    path = checkpoint_dir.expanduser().resolve()
    if (path / "pretrained_model").is_dir():
        path = path / "pretrained_model"
    if not (path / "config.json").is_file() or not (path / "model.safetensors").is_file():
        raise FileNotFoundError(
            f"Expected config.json and model.safetensors under {path}"
        )
    return path


def inspect_lerobot_checkpoint(checkpoint_dir: Path) -> dict[str, Any]:
    path = resolve_pretrained_model_dir(checkpoint_dir)
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    if config.get("type") != "act":
        raise ValueError(f"Expected ACT checkpoint, got {config.get('type')!r}")
    input_features = config.get("input_features", {})
    output_features = config.get("output_features", {})
    state_shape = _feature_shape(input_features, STATE_FEATURE)
    if state_shape != (STATE_DIM,):
        raise ValueError(
            f"ACT checkpoint uses state_dim={state_shape[0] if state_shape else 'unknown'}. "
            f"MobileMemBench evaluation requires {STATE_SCHEMA}."
        )
    action_shape = _feature_shape(output_features, ACTION_FEATURE)
    if action_shape != (13,):
        raise ValueError(f"ACT checkpoint must use action shape [13], got {action_shape}")
    camera_features = [
        key for key in input_features if str(key).startswith(CAMERA_PREFIX)
    ]
    if not camera_features:
        raise ValueError("ACT checkpoint has no visual input features")
    image_sizes = set()
    camera_features_by_alias = {}
    for key in camera_features:
        shape = _feature_shape(input_features, key)
        if len(shape) != 3 or shape[0] != 3:
            raise ValueError(f"ACT camera {key} must have CHW RGB shape, got {shape}")
        image_sizes.add((shape[1], shape[2]))
        dataset_name = str(key).removeprefix(CAMERA_PREFIX)
        alias = _semantic_camera_alias(dataset_name)
        if alias in camera_features_by_alias:
            raise ValueError(f"Duplicate ACT semantic camera alias: {alias}")
        camera_features_by_alias[alias] = str(key)
    required_aliases = {"agentview_left", "agentview_right", "eye_in_hand"}
    if set(camera_features_by_alias) != required_aliases:
        raise ValueError(
            "ACT checkpoint must contain exactly the three rgbd_v3 cameras: "
            + ", ".join(sorted(required_aliases))
        )
    if len(image_sizes) != 1:
        raise ValueError(f"ACT cameras use inconsistent image sizes: {sorted(image_sizes)}")
    metadata = read_checkpoint_metadata(path)
    state_schema = metadata.get("state_schema")
    if state_schema != STATE_SCHEMA:
        raise ValueError(
            f"ACT checkpoint must contain membench_policy.json with state_schema={STATE_SCHEMA}"
        )
    horizon = int(config.get("chunk_size", 0))
    if horizon <= 0:
        raise ValueError("ACT checkpoint chunk_size must be positive")
    return {
        "state_schema": state_schema,
        "camera_aliases": ["agentview_left", "agentview_right", "eye_in_hand"],
        "camera_features": camera_features_by_alias,
        "image_size": next(iter(image_sizes)),
        "horizon": horizon,
    }


def _feature_shape(features: Any, key: str) -> tuple[int, ...]:
    if not isinstance(features, dict) or key not in features:
        raise ValueError(f"Checkpoint feature missing: {key}")
    value = features[key]
    if not isinstance(value, dict) or not isinstance(value.get("shape"), list):
        raise ValueError(f"Checkpoint feature {key} has no shape")
    return tuple(int(item) for item in value["shape"])


def _semantic_camera_alias(dataset_name: str) -> str:
    aliases = {
        "robot0_agentview_right": "agentview_right",
        "robot0_agentview_left": "agentview_left",
        "robot0_eye_in_hand": "eye_in_hand",
        "agentview_right": "agentview_right",
        "agentview_left": "agentview_left",
        "eye_in_hand": "eye_in_hand",
    }
    try:
        return aliases[dataset_name]
    except KeyError as exc:
        raise ValueError(
            f"ACT checkpoint camera {dataset_name!r} has no MobileMemBench semantic alias"
        ) from exc
