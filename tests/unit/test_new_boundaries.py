from __future__ import annotations

import pytest

from bench.core import SeedStream, canonical_hash, stable_seed
from bench.execution.contracts import ExpertPlan, OperateStage, StageKind
from bench.execution.runner import StageRunner
from bench.recording.capture import CaptureBuffer
from bench.recording.validation import EpisodeDisposition, validate_aligned_lengths


def test_seed_and_hash_boundaries_are_stable() -> None:
    assert stable_seed(5, "scene") == stable_seed(5, "scene")
    assert stable_seed(5, "scene") != stable_seed(5, "objects")
    stream = SeedStream(5).iter("camera")
    assert (next(stream), next(stream)) == (SeedStream(5).child("camera", index=0), SeedStream(5).child("camera", index=1))
    assert canonical_hash({"b": 2, "a": 1}) == canonical_hash({"a": 1, "b": 2})


def test_expert_runner_short_circuits_and_reports_typed_stages() -> None:
    class Nav:
        def execute(self, target, **kwargs):
            return True

    class Manip:
        def execute(self, operation, target, **kwargs):
            return True

    plan = ExpertPlan(
        "TM01",
        (
            OperateStage(operation="noop", target="one", name="one"),
            OperateStage(operation="noop", target="two", name="two"),
        ),
    )
    result = StageRunner(navigation=Nav(), manipulation=Manip()).run(plan)
    assert result.success is True
    assert [item.kind for item in result.stages] == [StageKind.OPERATE, StageKind.OPERATE]


def test_capture_validation_distinguishes_error_and_rejected() -> None:
    buffer = CaptureBuffer()
    buffer.append(action=[0], observation={"x": 1}, timestamp=0.0)
    assert validate_aligned_lengths(buffer.fields()).disposition is EpisodeDisposition.ACCEPTED
    assert validate_aligned_lengths({"actions": [1], "observations": []}).disposition is EpisodeDisposition.REJECTED
    assert validate_aligned_lengths({}).disposition is EpisodeDisposition.ERROR
