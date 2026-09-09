"""MME-VLA Omega policy adapter for the canonical MemBench protocol."""

from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path

from bench.policy.backends.openpi import OpenPIBackend
from bench.policy.checkpoint import resolve_checkpoint_identity
from bench.runtime.data_contract import STATE_SCHEMA


class OmegaOpenPIBackend(OpenPIBackend):
    """Load the isolated Omega suite while reusing the OpenPI wire adapter."""

    @classmethod
    def from_checkpoint(
        cls,
        *,
        config_name: str,
        checkpoint_dir: Path,
        omega_repo: Path,
        omega_checkpoint: Path,
        omega_device: str = "cuda:0",
        image_size: tuple[int, int] = (224, 224),
        state_schema: str = STATE_SCHEMA,
        seed: int = 42,
        default_prompt: str | None = None,
        norm_asset_id: str | None = None,
        checkpoint_uri: str | None = None,
        server_id: str | None = None,
        code_revision: dict[str, str] | None = None,
    ) -> "OmegaOpenPIBackend":
        try:
            from mme_vla_omega_suite.policies import policy_config
            from mme_vla_omega_suite.training import config as training_config
        except ImportError as exc:
            raise RuntimeError(
                "MME-VLA Omega is not importable. Point MME_VLA_POLICY_SOURCE at "
                "policy/robomme_policy_learning/src on the omega-integration branch."
            ) from exc

        checkpoint_path = _required_directory(checkpoint_dir, "policy checkpoint")
        omega_repo_path = _required_directory(omega_repo, "VGGT-Omega repository")
        omega_checkpoint_path = _required_file(omega_checkpoint, "VGGT-Omega checkpoint")

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
        train_config = dataclasses.replace(
            train_config,
            omega_repo=str(omega_repo_path),
            omega_checkpoint=str(omega_checkpoint_path),
            omega_device=omega_device,
        )

        policy = policy_config.create_trained_policy(
            train_config,
            checkpoint_path,
            seed=seed,
            default_prompt=default_prompt,
        )
        backend = cls(
            policy,
            config_name=config_name,
            checkpoint=resolve_checkpoint_identity(
                checkpoint_path,
                uri=checkpoint_uri,
                step=_checkpoint_step(checkpoint_path),
            ),
            action_horizon=int(train_config.model.action_horizon),
            image_size=image_size,
            state_schema=state_schema,
            policy_cameras=tuple(train_config.model.omega_camera_order),
            server_id=server_id or f"omega-openpi-{config_name}",
            code_revision=code_revision,
        )
        backend._contract = dataclasses.replace(  # noqa: SLF001
            backend.contract,
            policy={
                "family": "mme_vla_omega",
                "name": config_name,
                "config": config_name,
                "omega_conditioning": train_config.model.omega_conditioning,
                "omega_memory_horizon": int(train_config.model.omega_memory_horizon),
                "omega_camera_order": list(train_config.model.omega_camera_order),
                "omega_checkpoint": str(omega_checkpoint_path),
                "omega_checkpoint_fingerprint": _fingerprint_file(omega_checkpoint_path),
            },
        )
        return backend


def _required_directory(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"{label} directory not found: {resolved}")
    return resolved


def _required_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} file not found: {resolved}")
    return resolved


def _checkpoint_step(path: Path) -> int | None:
    return int(path.name) if path.name.isdigit() else None


def _fingerprint_file(path: Path) -> str:
    """Fingerprint a large immutable weight file without reading all of it."""

    stat = path.stat()
    digest = hashlib.sha256()
    digest.update(path.name.encode())
    digest.update(b"\0")
    digest.update(str(stat.st_size).encode())
    digest.update(b"\0")
    with path.open("rb") as handle:
        digest.update(handle.read(64 * 1024))
        handle.seek(max(0, stat.st_size - 64 * 1024))
        digest.update(handle.read(64 * 1024))
    return f"sha256-manifest:{digest.hexdigest()}"


__all__ = ["OmegaOpenPIBackend"]
