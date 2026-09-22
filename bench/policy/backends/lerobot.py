"""LeRobot ACT and Diffusion backends with strict MemBench validation."""

from __future__ import annotations

import json
import logging
import random
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

from bench.evaluation.action import validate_action_chunk
from bench.policy.checkpoint import (
    read_checkpoint_metadata,
    resolve_checkpoint_identity,
)
from bench.policy.protocol import (
    ActionContract,
    InferenceRequest,
    InferenceResponse,
    ObservationContract,
    PolicyContract,
    RuntimeProfile,
    get_runtime_profile,
)
from bench.policy.protocol.profiles import SIM_RUNTIME_PROFILE

STATE_FEATURE = "observation.state"
ACTION_FEATURE = "action"
CAMERA_PREFIX = "observation.images."
LOGGER = logging.getLogger(__name__)


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
    ) -> LeRobotACTBackend:
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
                "runtime_profile": inspected["runtime_profile"],
                "camera_features": inspected["camera_features"],
            },
            checkpoint=resolve_checkpoint_identity(
                checkpoint_path,
                uri=checkpoint_uri,
                step=_checkpoint_step(checkpoint_path),
            ),
            observation=ObservationContract(
                state_schema=inspected["state_schema"],
                state_dim=inspected["state_dim"],
                required_cameras=tuple(inspected["camera_aliases"]),
                image_size=tuple(inspected["image_size"]),
            ),
            action=ActionContract(
                schema=inspected["action_schema"],
                dim=inspected["action_dim"],
                horizon=int(inspected["horizon"]),
            ),
            control_frequency_hz=inspected["control_frequency_hz"],
            runtime_profile=inspected["runtime_profile"],
            hardware_execution_semantics=inspected["hardware_execution_semantics"],
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
        get_runtime_profile(self.contract.runtime_profile)

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
            expected_dim=self.contract.action.dim,
            action_schema=self.contract.action.schema,
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


class LeRobotDiffusionBackend(LeRobotACTBackend):
    """Serve DiffusionPolicy through one-action wire responses.

    DiffusionPolicy.select_action must observe every environment step: it keeps
    the consecutive observation history required by n_obs_steps while serving
    cached actions from its native action chunk. The wire horizon is therefore
    one even though the policy's native action horizon is usually much larger.
    """

    def __init__(
        self,
        *,
        policy: Any,
        preprocessor: Any,
        postprocessor: Any,
        contract: PolicyContract,
        device: str,
        torch_module: Any,
        seed_by_episode: bool = True,
    ) -> None:
        super().__init__(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            contract=contract,
            device=device,
            torch_module=torch_module,
        )
        self.seed_by_episode = bool(seed_by_episode)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: Path,
        *,
        device: str = "cuda",
        checkpoint_uri: str | None = None,
        server_id: str | None = None,
        code_revision: dict[str, str] | None = None,
        seed_by_episode: bool = True,
    ) -> LeRobotDiffusionBackend:
        checkpoint_path = resolve_pretrained_model_dir(checkpoint_dir)
        inspected = inspect_lerobot_diffusion_checkpoint(checkpoint_path)
        try:
            import torch
            from lerobot.configs import PreTrainedConfig
            from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
            from lerobot.policies.factory import make_pre_post_processors
        except ImportError as exc:
            raise RuntimeError(
                "LeRobot with diffusion dependencies is not importable. Install "
                "the repository policy/lerobot environment with the diffusion extra."
            ) from exc

        policy_config = PreTrainedConfig.from_pretrained(checkpoint_path)
        if str(policy_config.type) != "diffusion":
            raise ValueError(
                "LeRobotDiffusionBackend requires a diffusion checkpoint, got "
                f"{policy_config.type!r}"
            )
        policy_config.device = device
        policy = DiffusionPolicy.from_pretrained(
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
            server_id=server_id or f"lerobot-diffusion-{checkpoint_path.parent.name}",
            policy={
                "family": "lerobot",
                "name": "diffusion",
                "config": "diffusion",
                "runtime_profile": inspected["runtime_profile"],
                "camera_features": inspected["camera_features"],
                "n_obs_steps": inspected["n_obs_steps"],
                "native_action_horizon": inspected["native_action_horizon"],
                "seed_policy": "episode_seed_v1" if seed_by_episode else "unseeded",
                "wire_action_mode": "select_action_v1",
            },
            checkpoint=resolve_checkpoint_identity(
                checkpoint_path,
                uri=checkpoint_uri,
                step=_checkpoint_step(checkpoint_path),
            ),
            observation=ObservationContract(
                state_schema=inspected["state_schema"],
                state_dim=inspected["state_dim"],
                required_cameras=tuple(inspected["camera_aliases"]),
                image_size=tuple(inspected["image_size"]),
            ),
            action=ActionContract(
                schema=inspected["action_schema"],
                dim=inspected["action_dim"],
                horizon=1,
            ),
            control_frequency_hz=inspected["control_frequency_hz"],
            runtime_profile=inspected["runtime_profile"],
            hardware_execution_semantics=inspected["hardware_execution_semantics"],
            code_revision=dict(code_revision or {}),
        )
        return cls(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            contract=contract,
            device=device,
            torch_module=torch,
            seed_by_episode=seed_by_episode,
        )

    def validate_checkpoint_contract(self) -> None:
        get_runtime_profile(self.contract.runtime_profile)
        if self.contract.action.horizon != 1:
            raise ValueError("LeRobot diffusion wire horizon must be one")

    def reset(self, episode_id: str) -> None:
        if self.seed_by_episode:
            seed = _episode_seed(episode_id)
            random.seed(seed)
            np.random.seed(seed % (2**32))
            self.torch.manual_seed(seed)
            LOGGER.info("Reset episode %s with diffusion seed %d", episode_id, seed)
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
            normalized_action = self.policy.select_action(processed)
            action = self.postprocessor(normalized_action)
        model_ms = (time.monotonic() - start) * 1000.0
        if hasattr(action, "detach"):
            action = action.detach().to(device="cpu", dtype=self.torch.float32).numpy()
        actions = np.asarray(action, dtype=np.float32)
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        actions = validate_action_chunk(
            actions,
            expected_horizon=1,
            expected_dim=self.contract.action.dim,
            action_schema=self.contract.action.schema,
        )
        return InferenceResponse(
            request_id=request.request_id,
            episode_id=request.episode_id,
            actions=actions,
            timing={"model_ms": model_ms},
        )


def resolve_pretrained_model_dir(checkpoint_dir: Path) -> Path:
    path = checkpoint_dir.expanduser().resolve()
    if (path / "pretrained_model").is_dir():
        path = path / "pretrained_model"
    if not (path / "config.json").is_file() or not (path / "model.safetensors").is_file():
        raise FileNotFoundError(f"Expected config.json and model.safetensors under {path}")
    return path


def inspect_lerobot_checkpoint(checkpoint_dir: Path) -> dict[str, Any]:
    path = resolve_pretrained_model_dir(checkpoint_dir)
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    if config.get("type") != "act":
        raise ValueError(f"Expected ACT checkpoint, got {config.get('type')!r}")
    input_features = config.get("input_features", {})
    output_features = config.get("output_features", {})
    state_shape = _feature_shape(input_features, STATE_FEATURE)
    action_shape = _feature_shape(output_features, ACTION_FEATURE)
    metadata = read_checkpoint_metadata(path)
    profile = _checkpoint_profile(
        metadata,
        state_dim=state_shape[0] if len(state_shape) == 1 else None,
        action_dim=action_shape[0] if len(action_shape) == 1 else None,
    )
    if state_shape != (profile.state_dim,):
        raise ValueError(
            f"ACT checkpoint uses state_dim={state_shape[0] if state_shape else 'unknown'}; "
            f"profile {profile.name} requires {profile.state_dim}"
        )
    if action_shape != (profile.action_dim,):
        raise ValueError(
            f"ACT checkpoint must use action shape [{profile.action_dim}], got {action_shape}"
        )
    camera_features_by_alias, image_size = _inspect_camera_features(
        input_features,
        metadata,
        profile,
        policy_label="ACT",
    )
    horizon = int(config.get("chunk_size", 0))
    if horizon <= 0:
        raise ValueError("ACT checkpoint chunk_size must be positive")
    return {
        "runtime_profile": profile.name,
        "state_schema": profile.state_schema,
        "state_dim": profile.state_dim,
        "action_schema": profile.action_schema,
        "action_dim": profile.action_dim,
        "camera_aliases": list(profile.required_cameras),
        "camera_features": camera_features_by_alias,
        "image_size": image_size,
        "horizon": horizon,
        "control_frequency_hz": profile.control_frequency_hz,
        "hardware_execution_semantics": metadata.get("hardware_execution_semantics"),
    }


def inspect_lerobot_diffusion_checkpoint(checkpoint_dir: Path) -> dict[str, Any]:
    path = resolve_pretrained_model_dir(checkpoint_dir)
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    if config.get("type") != "diffusion":
        raise ValueError(f"Expected diffusion checkpoint, got {config.get('type')!r}")
    common = _inspect_common_features(path, config, policy_label="diffusion")
    n_obs_steps = _positive_config_int(config, "n_obs_steps")
    horizon = _positive_config_int(config, "horizon")
    native_action_horizon = _positive_config_int(config, "n_action_steps")
    if native_action_horizon > horizon - n_obs_steps + 1:
        raise ValueError("Diffusion n_action_steps must be <= horizon - n_obs_steps + 1")
    metadata = read_checkpoint_metadata(path)
    if metadata.get("policy_type") not in (None, "diffusion"):
        raise ValueError("membench_policy.json policy_type does not match diffusion")
    return {
        **common,
        "n_obs_steps": n_obs_steps,
        "horizon": horizon,
        "native_action_horizon": native_action_horizon,
    }


def _inspect_common_features(
    path: Path,
    config: dict[str, Any],
    *,
    policy_label: str,
) -> dict[str, Any]:
    input_features = config.get("input_features", {})
    output_features = config.get("output_features", {})
    state_shape = _feature_shape(input_features, STATE_FEATURE)
    action_shape = _feature_shape(output_features, ACTION_FEATURE)
    metadata = read_checkpoint_metadata(path)
    profile = _checkpoint_profile(
        metadata,
        state_dim=state_shape[0] if len(state_shape) == 1 else None,
        action_dim=action_shape[0] if len(action_shape) == 1 else None,
    )
    if state_shape != (profile.state_dim,):
        raise ValueError(f"{policy_label} checkpoint state shape {state_shape} does not match profile")
    if action_shape != (profile.action_dim,):
        raise ValueError(
            f"{policy_label} checkpoint must use action shape [{profile.action_dim}], got {action_shape}"
        )
    camera_features, image_size = _inspect_camera_features(
        input_features,
        metadata,
        profile,
        policy_label=policy_label,
    )
    return {
        "runtime_profile": profile.name,
        "state_schema": profile.state_schema,
        "state_dim": profile.state_dim,
        "action_schema": profile.action_schema,
        "action_dim": profile.action_dim,
        "camera_aliases": list(profile.required_cameras),
        "camera_features": camera_features,
        "image_size": image_size,
        "control_frequency_hz": profile.control_frequency_hz,
        "hardware_execution_semantics": metadata.get("hardware_execution_semantics"),
    }


def _checkpoint_profile(
    metadata: dict[str, Any],
    *,
    state_dim: int | None,
    action_dim: int | None,
) -> RuntimeProfile:
    runtime_profile = metadata.get("runtime_profile")
    if runtime_profile is None:
        if metadata.get("state_schema") == "membench.robot_state37.v2":
            runtime_profile = SIM_RUNTIME_PROFILE
        else:
            raise ValueError(
                "Checkpoint must contain membench_policy.json with runtime_profile "
                "for non-simulation dimensions"
            )
    try:
        profile = get_runtime_profile(str(runtime_profile))
    except ValueError as exc:
        raise ValueError(str(exc)) from exc
    if metadata.get("state_schema") != profile.state_schema:
        raise ValueError(
            f"Checkpoint state_schema={metadata.get('state_schema')!r} does not match "
            f"profile {profile.name}"
        )
    if metadata.get("action_schema") not in (None, profile.action_schema):
        raise ValueError(
            f"Checkpoint action_schema={metadata.get('action_schema')!r} does not match "
            f"profile {profile.name}"
        )
    for key, actual, expected in (
        ("state_dim", metadata.get("state_dim"), profile.state_dim),
        ("action_dim", metadata.get("action_dim"), profile.action_dim),
    ):
        if actual is not None and actual != expected:
            raise ValueError(f"Checkpoint {key}={actual!r} does not match profile {profile.name}")
    is_simulation = profile.name == SIM_RUNTIME_PROFILE
    if not is_simulation:
        required = {
            "state_dim": profile.state_dim,
            "action_schema": profile.action_schema,
            "action_dim": profile.action_dim,
            "camera_aliases": list(profile.required_cameras),
            "image_size": list(profile.image_size),
            "control_frequency_hz": profile.control_frequency_hz,
            "hardware_execution_semantics": "unverified",
        }
        for key, expected in required.items():
            if metadata.get(key) != expected:
                raise ValueError(
                    f"Checkpoint {key}={metadata.get(key)!r} does not match "
                    f"profile value {expected!r}"
                )
    return profile


def _inspect_camera_features(
    input_features: Any,
    metadata: dict[str, Any],
    profile: RuntimeProfile,
    *,
    policy_label: str,
) -> tuple[dict[str, str], tuple[int, int]]:
    camera_keys = [key for key in input_features if str(key).startswith(CAMERA_PREFIX)]
    if not camera_keys:
        raise ValueError(f"{policy_label} checkpoint has no visual input features")
    image_sizes: set[tuple[int, int]] = set()
    for key in camera_keys:
        shape = _feature_shape(input_features, key)
        if len(shape) != 3 or shape[0] != 3:
            raise ValueError(f"{policy_label} camera {key} must have CHW RGB shape, got {shape}")
        image_sizes.add((shape[1], shape[2]))
    if len(image_sizes) != 1:
        raise ValueError(f"{policy_label} cameras use inconsistent image sizes: {sorted(image_sizes)}")
    image_size = next(iter(image_sizes))
    if image_size != profile.image_size:
        raise ValueError(
            f"{policy_label} cameras use image_size={image_size}; "
            f"profile {profile.name} requires {profile.image_size}"
        )

    raw_mapping = metadata.get("camera_mapping")
    if raw_mapping is not None:
        if not isinstance(raw_mapping, dict):
            raise TypeError("Checkpoint camera_mapping must be a map")
        camera_features = {str(alias): str(feature) for alias, feature in raw_mapping.items()}
    else:
        camera_features = {}
        for key in camera_keys:
            alias = _semantic_camera_alias(str(key).removeprefix(CAMERA_PREFIX))
            if alias in camera_features:
                raise ValueError(f"Duplicate {policy_label} semantic camera alias: {alias}")
            camera_features[alias] = str(key)
    if tuple(camera_features) != profile.required_cameras:
        raise ValueError(
            f"{policy_label} checkpoint camera aliases must be exactly "
            f"{list(profile.required_cameras)}, got {list(camera_features)}"
        )
    if set(camera_features.values()) != {str(key) for key in camera_keys}:
        raise ValueError("Checkpoint camera_mapping must reference every visual feature exactly once")
    return camera_features, image_size


def _positive_config_int(config: dict[str, Any], key: str) -> int:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"Diffusion checkpoint {key} must be a positive integer")
    return value


def _checkpoint_step(checkpoint_dir: Path) -> int | None:
    state_path = checkpoint_dir.parent / "training_state" / "training_step.json"
    if state_path.is_file():
        value = json.loads(state_path.read_text(encoding="utf-8")).get("step")
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"Invalid checkpoint training step in {state_path}")
        return value
    parent_name = checkpoint_dir.parent.name
    return int(parent_name) if parent_name.isdecimal() else None


def _episode_seed(episode_id: str) -> int:
    match = re.search(r"(?:^|-)seed-(\d+)(?:-|$)", episode_id)
    if match is None:
        raise ValueError(
            f"seed_by_episode requires an episode ID containing 'seed-<integer>': {episode_id!r}"
        )
    return int(match.group(1))


def _feature_shape(features: Any, key: str) -> tuple[int, ...]:
    if not isinstance(features, dict) or key not in features:
        raise ValueError(f"Checkpoint feature missing: {key}")
    value = features[key]
    if not isinstance(value, dict) or not isinstance(value.get("shape"), list):
        raise TypeError(f"Checkpoint feature {key} has no shape")
    return tuple(int(item) for item in value["shape"])


def _semantic_camera_alias(dataset_name: str) -> str:
    aliases = {
        "robot0_agentview_right": "agentview_right",
        "robot0_agentview_left": "agentview_left",
        "robot0_eye_in_hand": "eye_in_hand",
        "agentview_right": "agentview_right",
        "agentview_left": "agentview_left",
        "eye_in_hand": "eye_in_hand",
        "front": "front",
        "left": "left",
        "right": "right",
    }
    try:
        return aliases[dataset_name]
    except KeyError as exc:
        raise ValueError(
            f"ACT checkpoint camera {dataset_name!r} has no MobileMemBench semantic alias"
        ) from exc
