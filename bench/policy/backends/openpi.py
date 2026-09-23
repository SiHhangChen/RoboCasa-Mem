"""OpenPI pi0/pi0.5 backend for the MobileMemBench policy protocol."""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
import random
import re
import time
from typing import Any

import numpy as np

from bench.evaluation.action import validate_action_chunk
from bench.policy.protocol import (
    ActionContract,
    CheckpointIdentity,
    InferenceRequest,
    InferenceResponse,
    ObservationContract,
    PolicyContract,
)
from bench.policy.checkpoint import resolve_checkpoint_identity
from bench.policy.protocol.profiles import (
    COBOT_MAGIC_RUNTIME_PROFILE,
    SIM_RUNTIME_PROFILE,
    get_runtime_profile,
)
from bench.real_robot.wr07_prompts import FULL_CONFIG, SUBGOAL_CONFIG, prompt_policy

LOGGER = logging.getLogger(__name__)

# Fallback camera subset forwarded to the policy, used only when a config's
# repack does not declare an images mapping. The wire contract always requires
# the full 3-camera rgbd_v3 set; the backend forwards only the policy's subset.
_DEFAULT_POLICY_CAMERAS = ("agentview_right", "eye_in_hand")


def _policy_cameras_from_config(train_config: Any) -> tuple[str, ...]:
    """Derive the camera aliases a policy consumes from its config's repack.

    The repack maps dataset columns onto the image aliases fed to the policy, so
    its ``images`` keys are exactly the cameras the policy uses. This reads the
    pinned OpenPI config as-is (no fork changes required); the wire contract
    still requires all three rgbd_v3 cameras and the backend forwards only the
    derived subset to the policy.
    """
    repack = getattr(getattr(train_config, "data", None), "repack_transforms", None)
    if repack is not None:
        for transform in getattr(repack, "inputs", ()):
            structure = getattr(transform, "structure", None)
            if not isinstance(structure, dict):
                continue
            images = structure.get("images")
            if isinstance(images, dict) and images:
                return tuple(images.keys())
    return _DEFAULT_POLICY_CAMERAS


def _camera_map_from_config(train_config: Any, runtime_profile: str) -> dict[str, str]:
    policy_cameras = _policy_cameras_from_config(train_config)
    if runtime_profile == SIM_RUNTIME_PROFILE:
        return {alias: alias for alias in policy_cameras}
    if runtime_profile != COBOT_MAGIC_RUNTIME_PROFILE:
        raise ValueError(f"Unsupported OpenPI runtime profile: {runtime_profile}")
    inputs = train_config.data.repack_transforms.inputs
    images = next(
        (item.structure["images"] for item in inputs if "images" in item.structure), None
    )
    expected = {
        "agentview_left": "observation.images.front",
        "agentview_right": "observation.images.left",
        "eye_in_hand": "observation.images.right",
    }
    if images != expected:
        raise ValueError(f"WR07 OpenPI camera mapping differs from the trained configuration: {images}")
    return {
        model_alias: dataset_key.removeprefix("observation.images.")
        for model_alias, dataset_key in images.items()
    }


class OpenPIBackend:
    def __init__(
        self,
        policy: Any,
        *,
        config_name: str,
        checkpoint: CheckpointIdentity,
        action_horizon: int,
        image_size: tuple[int, int],
        state_schema: str | None = None,
        policy_cameras: tuple[str, ...] = _DEFAULT_POLICY_CAMERAS,
        camera_map: dict[str, str] | None = None,
        runtime_profile: str = SIM_RUNTIME_PROFILE,
        prompt_contract: dict[str, object] | None = None,
        server_id: str | None = None,
        code_revision: dict[str, str] | None = None,
        seed_by_episode: bool = False,
    ) -> None:
        self.policy = policy
        self.config_name = config_name
        self.seed_by_episode = bool(seed_by_episode)
        profile = get_runtime_profile(runtime_profile)
        camera_map = dict(
            {alias: alias for alias in policy_cameras} if camera_map is None else camera_map
        )
        if not camera_map or not set(camera_map.values()) <= set(profile.required_cameras):
            raise ValueError(
                f"policy cameras must map to a non-empty subset of {profile.required_cameras}, "
                f"got {camera_map}"
            )
        self._camera_map = camera_map
        self._prompt_contract = dict(prompt_contract or {})
        self._active_prompt: str | None = None
        if state_schema is not None and state_schema != profile.state_schema:
            raise ValueError(f"{runtime_profile} requires state schema {profile.state_schema}, got {state_schema}")
        self._contract = PolicyContract(
            server_id=server_id or f"openpi-{config_name}",
            policy={
                "family": "openpi",
                "name": config_name,
                "config": config_name,
                "seed_policy": "episode_seed_v1" if seed_by_episode else "unseeded",
                **self._prompt_contract,
            },
            checkpoint=checkpoint,
            observation=ObservationContract(
                state_schema=profile.state_schema,
                state_dim=profile.state_dim,
                required_cameras=profile.required_cameras,
                image_size=image_size,
            ),
            action=ActionContract(
                schema=profile.action_schema,
                dim=profile.action_dim,
                horizon=action_horizon,
            ),
            control_frequency_hz=profile.control_frequency_hz,
            runtime_profile=profile.name,
            hardware_execution_semantics=(
                "unverified" if profile.name == COBOT_MAGIC_RUNTIME_PROFILE else None
            ),
            code_revision=dict(code_revision or {}),
        )

    @classmethod
    def from_checkpoint(
        cls,
        *,
        config_name: str,
        checkpoint_dir: Path,
        image_size: tuple[int, int],
        state_schema: str | None = None,
        runtime_profile: str | None = None,
        default_prompt: str | None = None,
        pytorch_device: str | None = None,
        norm_asset_id: str | None = None,
        checkpoint_uri: str | None = None,
        server_id: str | None = None,
        code_revision: dict[str, str] | None = None,
        seed_by_episode: bool = False,
    ) -> "OpenPIBackend":
        try:
            from openpi.policies import policy_config
            from openpi.training import config as training_config
        except ImportError as exc:
            raise RuntimeError(
                "OpenPI is not importable. Run serve_openpi.py from this repository "
                "or install policy/openpi."
            ) from exc
        train_config = training_config.get_config(config_name)
        is_wr07_real = config_name in (FULL_CONFIG, SUBGOAL_CONFIG)
        expected_profile = COBOT_MAGIC_RUNTIME_PROFILE if is_wr07_real else SIM_RUNTIME_PROFILE
        if runtime_profile is not None and runtime_profile != expected_profile:
            raise ValueError(
                f"{config_name} requires runtime profile {expected_profile}, got {runtime_profile}"
            )
        runtime_profile = expected_profile
        profile = get_runtime_profile(runtime_profile)
        if state_schema is not None and state_schema != profile.state_schema:
            raise ValueError(
                f"{config_name} requires state schema {profile.state_schema}, got {state_schema}"
            )
        if is_wr07_real and (
            train_config.data.state_keep_dim != profile.state_dim
            or train_config.data.action_output_dim != profile.action_dim
        ):
            raise ValueError(
                f"{config_name} must use {profile.state_dim}D state and {profile.action_dim}D actions"
            )
        if norm_asset_id is not None:
            assets = dataclasses.replace(
                train_config.data.assets,
                asset_id=str(norm_asset_id),
            )
            train_config = dataclasses.replace(
                train_config,
                data=dataclasses.replace(train_config.data, assets=assets),
            )
        camera_map = _camera_map_from_config(train_config, runtime_profile)
        prompts = prompt_policy(config_name, default_prompt=train_config.data.default_prompt)
        horizon = int(train_config.model.action_horizon)
        checkpoint_path = checkpoint_dir.expanduser().resolve()
        policy = policy_config.create_trained_policy(
            train_config,
            checkpoint_path,
            default_prompt=default_prompt,
            pytorch_device=pytorch_device,
        )
        return cls(
            policy,
            config_name=config_name,
            checkpoint=resolve_checkpoint_identity(
                checkpoint_path,
                uri=checkpoint_uri,
            ),
            action_horizon=horizon,
            image_size=image_size,
            state_schema=profile.state_schema,
            camera_map=camera_map,
            runtime_profile=runtime_profile,
            prompt_contract=prompts,
            server_id=server_id,
            code_revision=code_revision,
            seed_by_episode=seed_by_episode,
        )

    @property
    def contract(self) -> PolicyContract:
        return self._contract

    def reset(self, episode_id: str) -> None:
        self._active_prompt = None
        if self.seed_by_episode:
            seed = _episode_seed(episode_id)
            random.seed(seed)
            np.random.seed(seed % (2**32))
            # torch.manual_seed also seeds the CUDA generators. Do this for
            # every episode, not just process startup, so resume is independent
            # of the number of previous requests or warmup calls.
            _reseed_policy(self.policy, seed)
            LOGGER.info("Reset episode %s with model seed %d", episode_id, seed)
        reset = getattr(self.policy, "reset", None)
        if callable(reset):
            reset()

    def infer(self, request: InferenceRequest) -> InferenceResponse:
        observation = request.observation
        prompt_mode = self._prompt_contract.get("prompt_mode")
        if prompt_mode == "fixed" and observation.prompt != self._prompt_contract["default_prompt"]:
            raise ValueError("Prompt differs from this checkpoint's fixed training prompt")
        if (
            prompt_mode == "episode_choice"
            and observation.prompt not in self._prompt_contract["prompt_choices"].values()
        ):
            raise ValueError("Prompt must be one of this checkpoint's three WR07 subgoal choices")
        if self._active_prompt is not None and observation.prompt != self._active_prompt:
            raise ValueError(
                "Prompt cannot change within an episode; reset before selecting another subgoal"
            )
        self._active_prompt = observation.prompt
        policy_input = {
            "images": {
                model_alias: observation.images[wire_alias]
                for model_alias, wire_alias in self._camera_map.items()
            },
            "state": observation.state,
            "prompt": observation.prompt,
        }
        start = time.monotonic()
        output = self.policy.infer(policy_input)
        elapsed_ms = (time.monotonic() - start) * 1000.0
        if not isinstance(output, dict) or "actions" not in output:
            raise ValueError("OpenPI policy output must contain actions")
        actions = validate_action_chunk(
            np.asarray(output["actions"]),
            expected_horizon=self.contract.action.horizon,
            expected_dim=self.contract.action.dim,
            action_schema=self.contract.action.schema,
        )
        policy_timing = output.get("policy_timing", {})
        model_ms = (
            float(policy_timing.get("infer_ms", elapsed_ms))
            if isinstance(policy_timing, dict)
            else elapsed_ms
        )
        return InferenceResponse(
            request_id=request.request_id,
            episode_id=request.episode_id,
            actions=actions,
            timing={"model_ms": model_ms},
        )

    def close(self) -> None:
        close = getattr(self.policy, "close", None)
        if callable(close):
            close()


def _reseed_policy(policy: Any, seed: int) -> None:
    """Restart the policy's sampling RNG from ``seed`` when it supports it."""
    reseed = getattr(policy, "reseed", None)
    if callable(reseed):
        reseed(seed)
        return
    # PyTorch-based OpenPI policies sample with the global torch generators.
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        LOGGER.warning("Cannot reseed policy RNG: policy has no reseed() and torch is unavailable")


def _episode_seed(episode_id: str) -> int:
    match = re.search(r"(?:^|-)seed-(\d+)(?:-|$)", episode_id)
    if match is None:
        raise ValueError(
            f"seed_by_episode requires an episode ID containing 'seed-<integer>': {episode_id!r}"
        )
    return int(match.group(1))
