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
from bench.runtime.data_contract import ACTION_DIM, ACTION_SCHEMA, STATE_DIM, STATE_SCHEMA

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


class OpenPIBackend:
    def __init__(
        self,
        policy: Any,
        *,
        config_name: str,
        checkpoint: CheckpointIdentity,
        action_horizon: int,
        image_size: tuple[int, int],
        state_schema: str = STATE_SCHEMA,
        policy_cameras: tuple[str, ...] = _DEFAULT_POLICY_CAMERAS,
        server_id: str | None = None,
        code_revision: dict[str, str] | None = None,
        seed_by_episode: bool = False,
    ) -> None:
        self.policy = policy
        self.config_name = config_name
        self.seed_by_episode = bool(seed_by_episode)
        policy_cameras = tuple(policy_cameras)
        required_cameras = ("agentview_left", "agentview_right", "eye_in_hand")
        if not policy_cameras or not set(policy_cameras) <= set(required_cameras):
            raise ValueError(
                f"policy_cameras must be a non-empty subset of {required_cameras}, "
                f"got {policy_cameras}"
            )
        self._policy_cameras = policy_cameras
        self._contract = PolicyContract(
            server_id=server_id or f"openpi-{config_name}",
            policy={
                "family": "openpi",
                "name": config_name,
                "config": config_name,
                "seed_policy": "episode_seed_v1" if seed_by_episode else "unseeded",
            },
            checkpoint=checkpoint,
            observation=ObservationContract(
                state_schema=state_schema,
                state_dim=STATE_DIM,
                required_cameras=required_cameras,
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
        config_name: str,
        checkpoint_dir: Path,
        image_size: tuple[int, int],
        state_schema: str,
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
        if norm_asset_id is not None:
            assets = dataclasses.replace(
                train_config.data.assets,
                asset_id=str(norm_asset_id),
            )
            train_config = dataclasses.replace(
                train_config,
                data=dataclasses.replace(train_config.data, assets=assets),
            )
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
            state_schema=state_schema,
            policy_cameras=_policy_cameras_from_config(train_config),
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
            _reseed_policy(self.policy, seed)
            LOGGER.info("Reset episode %s with model seed %d", episode_id, seed)
        reset = getattr(self.policy, "reset", None)
        if callable(reset):
            reset()

    def infer(self, request: InferenceRequest) -> InferenceResponse:
        observation = request.observation
        policy_input = {
            "images": {
                alias: observation.images[alias] for alias in self._policy_cameras
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
