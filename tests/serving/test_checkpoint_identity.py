from __future__ import annotations

from pathlib import Path

from bench.policy.checkpoint import (
    fingerprint_checkpoint,
    resolve_checkpoint_identity,
)


def test_checkpoint_manifest_fingerprint_is_stable_and_content_sensitive(
    tmp_path: Path,
):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text('{"step": 10}\n', encoding="utf-8")
    (checkpoint / "weights.bin").write_bytes(b"weights")

    first = fingerprint_checkpoint(checkpoint)
    second = fingerprint_checkpoint(checkpoint)
    identity = resolve_checkpoint_identity(
        checkpoint,
        uri="checkpoint://act/wx01/10",
    )

    assert first == second == identity.fingerprint
    assert first.startswith("sha256-manifest:")
    assert identity.uri == "checkpoint://act/wx01/10"

    (checkpoint / "weights.bin").write_bytes(b"changed")
    assert fingerprint_checkpoint(checkpoint) != first
