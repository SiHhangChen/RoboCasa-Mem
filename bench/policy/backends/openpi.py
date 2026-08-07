"""OpenPI pi0/pi0.5 backend for the MobileMemBench policy protocol."""

from __future__ import annotations

from pathlib import Path
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
    ) -> None:
        self.policy = policy
        self.config_name = config_name
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
        checkpoint_uri: str | None = None,
        server_id: str | None = None,
        code_revision: dict[str, str] | None = None,
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
        )

    @property
    def contract(self) -> PolicyContract:
        return self._contract

    def reset(self, episode_id: str) -> None:
        del episode_id
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
