from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bench.policy.backends.memer_openpi import MemEROpenPIBackend
from bench.policy.protocol import (
    ActionContract,
    CanonicalObservation,
    CheckpointIdentity,
    InferenceRequest,
    InferenceResponse,
    ObservationContract,
    PolicyContract,
)
from bench.runtime.data_contract import ACTION_SCHEMA, STATE_SCHEMA


@dataclass
class _HighLevelResult:
    current_subtask: str | None
    parse_ok: bool = True
    memory_indices_after: tuple[int, ...] = ()


class _FakeMemER:
    def __init__(self, results: list[_HighLevelResult]) -> None:
        self.results = list(results)
        self.reset_instructions: list[str | None] = []
        self.observations: list[dict[str, np.ndarray]] = []

    def reset(self, instruction: str | None = None) -> None:
        self.reset_instructions.append(instruction)

    def step(self, observation):
        self.observations.append(observation)
        return self.results.pop(0)


class _FakeLowLevel:
    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.reset_ids: list[str] = []
        self.closed = False
        self.contract = PolicyContract(
            server_id="fake-openpi",
            policy={"family": "openpi", "name": "fake"},
            checkpoint=CheckpointIdentity(uri="file://fake", fingerprint="sha256:fake"),
            observation=ObservationContract(
                state_schema=STATE_SCHEMA,
                state_dim=37,
                required_cameras=("agentview_left", "agentview_right", "eye_in_hand"),
                image_size=(224, 224),
            ),
            action=ActionContract(schema=ACTION_SCHEMA, dim=13, horizon=50),
            control_frequency_hz=20.0,
        )

    def reset(self, episode_id: str) -> None:
        self.reset_ids.append(episode_id)

    def infer(self, request: InferenceRequest) -> InferenceResponse:
        self.prompts.append(request.observation.prompt)
        return InferenceResponse(
            request_id=request.request_id,
            episode_id=request.episode_id,
            actions=np.zeros((50, 13), dtype=np.float32),
            timing={"model_ms": 2.0},
        )

    def close(self) -> None:
        self.closed = True


def _request(step: int = 0) -> InferenceRequest:
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    return InferenceRequest(
        request_id=f"request-{step}",
        episode_id="episode-1",
        step=step,
        observation=CanonicalObservation(
            task_id="WA-01",
            state=np.zeros(37, dtype=np.float32),
            images={
                "agentview_left": image,
                "agentview_right": image,
                "eye_in_hand": image,
            },
            prompt="full task prompt",
        ),
    )


def test_memer_subtask_replaces_the_low_level_prompt() -> None:
    high = _FakeMemER(
        [_HighLevelResult("add one sugar cube to a coffee", memory_indices_after=(4,))]
    )
    low = _FakeLowLevel()
    backend = MemEROpenPIBackend(
        high,
        low,
        instruction="full task prompt",
        allowed_subtasks=("add one sugar cube to a coffee",),
    )

    backend.reset("episode-1")
    response = backend.infer(_request())

    assert low.prompts == ["add one sugar cube to a coffee"]
    assert response.actions.shape == (50, 13)
    assert response.timing["subtask_fallback"] == 0.0
    assert response.timing["subtask_memory_count"] == 1.0
    assert set(high.observations[0]) == {
        "observation.images.robot0_agentview_left",
        "observation.images.robot0_agentview_right",
        "observation.images.robot0_eye_in_hand",
    }


def test_invalid_prediction_reuses_the_last_valid_subtask() -> None:
    high = _FakeMemER(
        [
            _HighLevelResult("carry the first coffee to the dining counter"),
            _HighLevelResult("not a trained label", parse_ok=True),
        ]
    )
    low = _FakeLowLevel()
    backend = MemEROpenPIBackend(
        high,
        low,
        instruction="full task prompt",
        allowed_subtasks=("carry the first coffee to the dining counter",),
    )

    backend.reset("episode-1")
    backend.infer(_request(0))
    response = backend.infer(_request(10))

    assert low.prompts == [
        "carry the first coffee to the dining counter",
        "carry the first coffee to the dining counter",
    ]
    assert response.timing["subtask_fallback"] == 1.0


def test_first_parse_failure_falls_back_to_the_first_trained_subtask() -> None:
    high = _FakeMemER([_HighLevelResult(None, parse_ok=False)])
    low = _FakeLowLevel()
    backend = MemEROpenPIBackend(
        high,
        low,
        instruction="full task prompt",
        allowed_subtasks=(
            "add one sugar cube to a coffee",
            "identify the sugared coffee",
        ),
    )

    backend.reset("episode-1")
    response = backend.infer(_request())

    assert low.prompts == ["add one sugar cube to a coffee"]
    assert response.timing["subtask_parse_ok"] == 0.0
    assert response.timing["subtask_fallback"] == 1.0
    assert response.timing["subtask_index"] == 0.0
