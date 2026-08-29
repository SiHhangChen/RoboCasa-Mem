"""Stage-conditioned policy evaluation with an evaluator-only handoff.

This module deliberately knows nothing about RoboCasa, a policy backend, or a
task's private simulator state.  A task supplies a small port exposing
observation, privileged-state, and step operations; the evaluator keeps the
privileged state and control source out of the model input.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
import inspect
from statistics import NormalDist
from typing import Any, Protocol, TypeAlias

import numpy as np


SUCCESS = "SUCCESS"
FAILURE = "FAILURE"
INVALID_PREFIX = "INVALID_PREFIX"


class StageStatus(str, Enum):
    SUCCESS = SUCCESS
    FAILURE = FAILURE
    INVALID_PREFIX = INVALID_PREFIX

Predicate: TypeAlias = Callable[[Any], bool]


def _false(_: Any) -> bool:
    return False


@dataclass(frozen=True, slots=True)
class StageSpec:
    """Evaluator-owned definition of one sequential task stage."""

    id: str
    completion_predicate: Predicate
    failure_predicate: Predicate = _false
    entry_predicate: Predicate | None = None
    model_step_budget: int = 1
    success_hold_steps: int = 1

    def __post_init__(self) -> None:
        if not str(self.id).strip():
            raise ValueError("stage id must be non-empty")
        if not callable(self.completion_predicate):
            raise TypeError(f"stage {self.id!r} completion_predicate must be callable")
        if not callable(self.failure_predicate):
            raise TypeError(f"stage {self.id!r} failure_predicate must be callable")
        if self.entry_predicate is not None and not callable(self.entry_predicate):
            raise TypeError(f"stage {self.id!r} entry_predicate must be callable")
        if int(self.model_step_budget) <= 0:
            raise ValueError(f"stage {self.id!r} model_step_budget must be positive")
        if int(self.success_hold_steps) <= 0:
            raise ValueError(f"stage {self.id!r} success_hold_steps must be positive")
        object.__setattr__(self, "id", str(self.id))
        object.__setattr__(self, "model_step_budget", int(self.model_step_budget))
        object.__setattr__(self, "success_hold_steps", int(self.success_hold_steps))


class StageTask(Protocol):
    """Minimal environment port required by :func:`evaluate_stage`."""

    task_id: str
    instruction: str
    stage_specs: Sequence[StageSpec]

    def reset(self, seed: int) -> Any: ...

    def observe(self) -> Any: ...

    def get_privileged_state(self) -> Any: ...

    def step(self, action: Any) -> Any: ...

    def close(self) -> None: ...


class ModelAdapter(Protocol):
    """The only policy interface visible to the stage evaluator."""

    def reset_episode(self, instruction: str) -> None: ...

    def predict(self, observation: Any) -> Any: ...


class ExpertController(Protocol):
    """Evaluator-side expert port; it receives privileged state only."""

    def predict(self, privileged_state: Any) -> Any: ...


# More descriptive spelling for integrations that prefer a concrete role name.
StageEvaluationTask = StageTask


@dataclass(slots=True)
class FunctionalStageTask:
    """Convenience task port for simulator-free tests and small integrations."""

    task_id: str
    instruction: str
    stage_specs: tuple[StageSpec, ...]
    reset_fn: Callable[[int], Any]
    observe_fn: Callable[[], Any]
    privileged_state_fn: Callable[[], Any]
    step_fn: Callable[[Any], Any]
    close_fn: Callable[[], None] = lambda: None

    def reset(self, seed: int) -> Any:
        return self.reset_fn(int(seed))

    def observe(self) -> Any:
        return self.observe_fn()

    def get_privileged_state(self) -> Any:
        return self.privileged_state_fn()

    def step(self, action: Any) -> Any:
        return self.step_fn(action)

    def close(self) -> None:
        self.close_fn()


@dataclass(slots=True)
class StageTracker:
    """Latch stage completion and failure using evaluator-private state only."""

    stages: tuple[StageSpec, ...]
    _completed: dict[str, bool] = field(init=False, repr=False)
    _failed: dict[str, bool] = field(init=False, repr=False)
    _streaks: dict[str, int] = field(init=False, repr=False)
    _newly_completed: tuple[str, ...] = field(init=False, default=(), repr=False)
    _last_state: Any = field(init=False, default=None, repr=False)
    _last_transition: Any = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        self.stages = tuple(self.stages)
        if not self.stages:
            raise ValueError("at least one stage is required")
        ids = [stage.id for stage in self.stages]
        if len(ids) != len(set(ids)):
            raise ValueError("stage ids must be unique")
        self._completed = {}
        self._failed = {}
        self._streaks = {}
        self._completed = {stage.id: False for stage in self.stages}
        self._failed = {stage.id: False for stage in self.stages}
        self._streaks = {stage.id: 0 for stage in self.stages}
        self._newly_completed = ()

    @property
    def stage_ids(self) -> tuple[str, ...]:
        return tuple(stage.id for stage in self.stages)

    @property
    def current_stage(self) -> str | None:
        for stage in self.stages:
            if not self._completed[stage.id]:
                return stage.id
        return self.stages[-1].id if self.stages else None

    def reset(self, privileged_state: Any) -> None:
        self._completed = {stage.id: False for stage in self.stages}
        self._failed = {stage.id: False for stage in self.stages}
        self._streaks = {stage.id: 0 for stage in self.stages}
        self._last_state = privileged_state
        self._last_transition = None
        # Initial state is part of the evaluator timeline.  This catches an
        # accidentally pre-completed target instead of silently counting it.
        self._evaluate(privileged_state)

    def update(self, privileged_state: Any, transition: Any = None) -> None:
        self._last_state = privileged_state
        self._last_transition = transition
        self._evaluate(privileged_state)

    @property
    def newly_completed_stages(self) -> tuple[str, ...]:
        """Stages latched by the most recent reset/update call."""

        return self._newly_completed

    def prefix_completed(self, target_stage: str | int) -> bool:
        index = self._index(target_stage)
        if index == 0:
            return True
        if not all(self._completed[stage.id] for stage in self.stages[:index]):
            return False
        entry = self.stages[index].entry_predicate
        return entry is None or bool(entry(self._last_state))

    def target_entry_reached(self, target_stage: str | int) -> bool:
        """Return whether the evaluator may hand control to this stage."""

        return self.prefix_completed(target_stage)

    def target_already_completed(self, target_stage: str | int) -> bool:
        return bool(self._completed[self._stage(target_stage).id])

    def target_completed(self, target_stage: str | int) -> bool:
        return self.target_already_completed(target_stage)

    def target_failed(self, target_stage: str | int) -> bool:
        return bool(self._failed[self._stage(target_stage).id])

    def expert_prefix_failed(self, target_stage: str | int | None = None) -> bool:
        end = len(self.stages) if target_stage is None else self._index(target_stage)
        return any(self._failed[stage.id] for stage in self.stages[:end])

    def snapshot(self) -> dict[str, Any]:
        return {
            "current_stage": self.current_stage,
            "completed": dict(self._completed),
            "failed": dict(self._failed),
            "completion_streaks": dict(self._streaks),
        }

    def _evaluate(self, state: Any) -> None:
        before = set(stage.id for stage in self.stages if self._completed[stage.id])
        for stage in self.stages:
            if self._failed[stage.id]:
                continue
            if bool(stage.failure_predicate(state)):
                self._failed[stage.id] = True
                continue
            if self._completed[stage.id]:
                continue
            if bool(stage.completion_predicate(state)):
                self._streaks[stage.id] += 1
                if self._streaks[stage.id] >= stage.success_hold_steps:
                    self._completed[stage.id] = True
            else:
                self._streaks[stage.id] = 0
        self._newly_completed = tuple(
            stage.id
            for stage in self.stages
            if self._completed[stage.id] and stage.id not in before
        )

    def _stage(self, value: str | int) -> StageSpec:
        return self.stages[self._index(value)]

    def _index(self, value: str | int) -> int:
        if isinstance(value, int) and not isinstance(value, bool):
            if value < 0 or value >= len(self.stages):
                raise IndexError(f"stage index out of range: {value}")
            return value
        rendered = str(value)
        for index, stage in enumerate(self.stages):
            if stage.id == rendered:
                return index
        raise KeyError(f"unknown stage {value!r}; expected one of {self.stage_ids}")


@dataclass(frozen=True, slots=True)
class StageStepLog:
    """One control transition.  Fields marked private must never be model input."""

    task_id: str
    episode_seed: int
    target_stage: str
    environment_step: int
    model_inference_index: int
    observation_id: int
    observation_timestamp: float | None
    privileged_stage_state: Any
    proposed_model_action: Any
    executed_action: Any
    action_source: str
    expert_prefix_valid: bool
    target_success: bool
    target_failure: bool
    model_control_steps: int
    environment_done: bool
    stage: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "episode_seed": self.episode_seed,
            "target_stage": self.target_stage,
            "environment_step": self.environment_step,
            "model_inference_index": self.model_inference_index,
            "observation_id": self.observation_id,
            "observation_timestamp": self.observation_timestamp,
            "privileged_stage_state": _jsonable(self.privileged_stage_state),
            "proposed_model_action": _jsonable(self.proposed_model_action),
            "executed_action": _jsonable(self.executed_action),
            "action_source": self.action_source,
            "expert_prefix_valid": self.expert_prefix_valid,
            "target_success": self.target_success,
            "target_failure": self.target_failure,
            "model_control_steps": self.model_control_steps,
            "environment_done": self.environment_done,
            "stage": self.stage,
        }


@dataclass(slots=True)
class StageEvaluationResult:
    task_id: str
    target_stage: str
    seed: int
    status: str
    success: bool
    success_step: int | None
    expert_prefix_steps: int
    model_control_steps: int
    total_model_inference_calls: int
    environment_steps: int
    handoff_observation_id: int | None
    failure_reason: str | None = None
    logs: tuple[StageStepLog, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in {SUCCESS, FAILURE, INVALID_PREFIX}:
            raise ValueError(f"unsupported stage result status: {self.status!r}")
        if bool(self.success) != (self.status == SUCCESS):
            raise ValueError("stage result status and success disagree")
        if self.status == SUCCESS and self.success_step is None:
            raise ValueError("successful stage result must include success_step")
        counters = (
            self.expert_prefix_steps,
            self.model_control_steps,
            self.total_model_inference_calls,
            self.environment_steps,
        )
        if any(int(value) < 0 for value in counters):
            raise ValueError("stage result counters must be non-negative")

    @property
    def valid_prefix(self) -> bool:
        return self.status != INVALID_PREFIX

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "target_stage": self.target_stage,
            "seed": self.seed,
            "status": self.status,
            "success": self.success,
            "success_step": self.success_step,
            "expert_prefix_steps": self.expert_prefix_steps,
            "model_control_steps": self.model_control_steps,
            "total_model_inference_calls": self.total_model_inference_calls,
            "environment_steps": self.environment_steps,
            "handoff_observation_id": self.handoff_observation_id,
            "failure_reason": self.failure_reason,
            "logs": [item.to_dict() for item in self.logs],
        }


@dataclass(frozen=True, slots=True)
class StageStatistics:
    """Counts and rates for one task/stage, excluding invalid prefixes."""

    task_id: str
    target_stage: str
    successes: int
    failures: int
    invalid_prefixes: int
    attempted: int

    def __post_init__(self) -> None:
        values = (
            self.successes,
            self.failures,
            self.invalid_prefixes,
            self.attempted,
        )
        if any(int(value) < 0 for value in values):
            raise ValueError("stage statistics counts must be non-negative")
        if int(self.attempted) != int(
            self.successes + self.failures + self.invalid_prefixes
        ):
            raise ValueError("attempted must equal successes + failures + invalid_prefixes")

    @property
    def valid_tests(self) -> int:
        return self.successes + self.failures

    @property
    def stage_sr(self) -> float | None:
        return _rate(self.successes, self.valid_tests)

    @property
    def prefix_validity(self) -> float | None:
        return _rate(self.valid_tests, self.attempted)

    def wilson_interval(self, confidence: float = 0.95) -> tuple[float, float] | None:
        return wilson_interval(self.successes, self.valid_tests, confidence=confidence)

    def to_dict(self, *, confidence: float = 0.95) -> dict[str, Any]:
        interval = self.wilson_interval(confidence)
        return {
            "task_id": self.task_id,
            "target_stage": self.target_stage,
            "successes": self.successes,
            "failures": self.failures,
            "invalid_prefixes": self.invalid_prefixes,
            "attempted": self.attempted,
            "valid_tests": self.valid_tests,
            "stage_sr": self.stage_sr,
            "prefix_validity": self.prefix_validity,
            "confidence_interval": None if interval is None else list(interval),
        }


def aggregate_stage_results(
    results: Sequence[StageEvaluationResult],
    *,
    task_id: str | None = None,
    target_stage: str | None = None,
) -> StageStatistics:
    """Aggregate results using ``success + failure`` as the StageSR denominator."""

    selected = [
        result
        for result in results
        if (task_id is None or result.task_id == task_id)
        and (target_stage is None or result.target_stage == target_stage)
    ]
    if not selected:
        raise ValueError("cannot aggregate an empty stage result set")
    task_values = {result.task_id for result in selected}
    stage_values = {result.target_stage for result in selected}
    return StageStatistics(
        task_id=next(iter(task_values)) if len(task_values) == 1 else "*",
        target_stage=next(iter(stage_values)) if len(stage_values) == 1 else "*",
        successes=sum(result.status == SUCCESS for result in selected),
        failures=sum(result.status == FAILURE for result in selected),
        invalid_prefixes=sum(result.status == INVALID_PREFIX for result in selected),
        attempted=len(selected),
    )


def stage_success_rate(results: Sequence[StageEvaluationResult]) -> float | None:
    """Return StageSR, with invalid expert prefixes omitted from the denominator."""

    return aggregate_stage_results(results).stage_sr


def stage_sr(results: Sequence[StageEvaluationResult]) -> float | None:
    """Short alias matching the metric notation used in benchmark reports."""

    return stage_success_rate(results)


def prefix_validity(results: Sequence[StageEvaluationResult]) -> float | None:
    return aggregate_stage_results(results).prefix_validity


def macro_stage_sr(results: Sequence[StageEvaluationResult]) -> float | None:
    """Average stage rates within each task, then average tasks equally."""

    by_task: dict[str, list[StageEvaluationResult]] = {}
    for result in results:
        by_task.setdefault(result.task_id, []).append(result)
    task_means: list[float] = []
    for task_results in by_task.values():
        by_stage: dict[str, list[StageEvaluationResult]] = {}
        for result in task_results:
            by_stage.setdefault(result.target_stage, []).append(result)
        stage_rates = [
            rate
            for rate in (stage_success_rate(values) for values in by_stage.values())
            if rate is not None
        ]
        if stage_rates:
            task_means.append(sum(stage_rates) / len(stage_rates))
    return None if not task_means else sum(task_means) / len(task_means)


def wilson_interval(
    successes: int,
    trials: int,
    *,
    confidence: float = 0.95,
) -> tuple[float, float] | None:
    """Wilson score interval for a Bernoulli rate."""

    if trials <= 0:
        return None
    if successes < 0 or successes > trials:
        raise ValueError("successes must be between zero and trials")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    # ``statistics`` keeps the evaluator lightweight while supporting any
    # confidence level in (0, 1), rather than hard-coding only 95%.
    z = NormalDist().inv_cdf(0.5 + float(confidence) / 2.0)
    p = successes / trials
    denominator = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denominator
    radius = z * np.sqrt(p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def evaluate_stage(
    task: StageTask,
    target_stage: str | int,
    seed: int,
    model: Any,
    expert: Any,
    *,
    tracker: StageTracker | None = None,
    expert_prefix_budget: int | None = None,
    record_logs: bool = True,
    logger: Callable[[Mapping[str, Any]], None] | Any | None = None,
) -> StageEvaluationResult:
    """Evaluate exactly one stage after an online expert-controlled prefix.

    ``model.predict`` is called with observations only.  The evaluator never
    sends privileged state, expert actions, stage ids, or a handoff flag to the
    model.  The action used on the first target-stage transition is predicted
    from the observation produced by the prefix's final transition.
    """

    stages = _task_stages(task)
    tracker = tracker or StageTracker(stages)
    target_index = tracker._index(target_stage)
    target_id = tracker.stages[target_index].id
    if expert_prefix_budget is None:
        configured_prefix_budget = getattr(task, "expert_prefix_budget", None)
        if configured_prefix_budget is not None:
            expert_prefix_budget = int(configured_prefix_budget)
    instruction = _task_instruction(task)
    reset_episode = getattr(model, "reset_episode", None)
    predict = getattr(model, "predict", None)
    expert_predict = getattr(expert, "predict", None)
    if not callable(reset_episode) or not callable(predict):
        raise TypeError("model must expose reset_episode(instruction) and predict(observation)")
    if target_index > 0 and not callable(expert_predict):
        raise TypeError("expert must expose predict(privileged_state)")

    try:
        reset_value = _reset_task(task, int(seed))
        reset_episode(instruction)
        tracker.reset(_privileged_state(task))
        observation = _observe_task(task, reset_value)
    except BaseException:
        _close_task(task)
        raise

    model_inference_calls = 0
    env_steps = 0
    expert_steps = 0
    model_steps = 0
    success_step: int | None = None
    handoff_observation_id: int | None = None
    handed_off = target_index == 0
    action_buffer: list[Any] = []
    logs: list[StageStepLog] = []

    # A pre-completed target is never a policy success.  This also catches an
    # expert action that completes both the prefix and target in one transition.
    if tracker.target_already_completed(target_id) or tracker.target_failed(target_id):
        _close_task(task)
        return _result(
            task, target_id, seed, INVALID_PREFIX, False, None, expert_steps,
            model_steps, model_inference_calls, env_steps, handoff_observation_id,
            (
                "target_already_completed_at_reset"
                if tracker.target_already_completed(target_id)
                else "target_failed_at_reset"
            ),
            logs,
        )

    initial_prefix_stages = {
        stage.id for stage in tracker.stages[:target_index]
    }.intersection(tracker.newly_completed_stages)
    if len(initial_prefix_stages) > 1:
        _close_task(task)
        return _result(
            task, target_id, seed, INVALID_PREFIX, False, None, expert_steps,
            model_steps, model_inference_calls, env_steps, handoff_observation_id,
            "expert_skipped_stage_boundary_at_reset", logs,
        )

    try:
        while True:
            if not handed_off:
                if tracker.target_already_completed(target_id):
                    return _result(
                        task, target_id, seed, INVALID_PREFIX, False, None, expert_steps,
                        model_steps, model_inference_calls, env_steps, handoff_observation_id,
                        "expert_completed_target_during_prefix", logs,
                    )
                if tracker.target_failed(target_id):
                    return _result(
                        task, target_id, seed, INVALID_PREFIX, False, None, expert_steps,
                        model_steps, model_inference_calls, env_steps, handoff_observation_id,
                        "expert_target_failure_during_prefix", logs,
                    )
                if tracker.prefix_completed(target_id):
                    handed_off = True
                    handoff_observation_id = env_steps
                    # Any chunk proposed before handoff is intentionally lost.
                    action_buffer.clear()

            if handed_off and tracker.target_already_completed(target_id):
                return _result(
                    task, target_id, seed, SUCCESS, True, success_step, expert_steps,
                    model_steps, model_inference_calls, env_steps, handoff_observation_id,
                    None, logs,
                )

            if not handed_off and tracker.expert_prefix_failed(target_id):
                return _result(
                    task, target_id, seed, INVALID_PREFIX, False, None, expert_steps,
                    model_steps, model_inference_calls, env_steps, handoff_observation_id,
                    "expert_prefix_failed", logs,
                )

            if handed_off and model_steps >= tracker.stages[target_index].model_step_budget:
                return _result(
                    task, target_id, seed, FAILURE, False, success_step, expert_steps,
                    model_steps, model_inference_calls, env_steps, handoff_observation_id,
                    "model_step_budget", logs,
                )
            if (
                not handed_off
                and expert_prefix_budget is not None
                and expert_steps >= int(expert_prefix_budget)
            ):
                return _result(
                    task, target_id, seed, INVALID_PREFIX, False, None, expert_steps,
                    model_steps, model_inference_calls, env_steps, handoff_observation_id,
                    "expert_prefix_budget", logs,
                )

            privileged = _privileged_state(task)
            if handed_off and action_buffer:
                proposed = action_buffer.pop(0)
                inference_index = model_inference_calls - 1
            else:
                # Prefix chunks are discarded in full.  Handoff always starts
                # with a fresh prediction from the current observation.
                proposed = predict(observation)
                model_inference_calls += 1
                if handed_off:
                    rows = _action_rows(proposed)
                    proposed_action = rows.pop(0)
                    action_buffer.extend(rows)
                    proposed = proposed_action
                inference_index = model_inference_calls - 1

            if handed_off:
                executed = proposed
                action_source = "model"
                model_steps += 1
            else:
                executed = expert_predict(privileged)
                action_source = "expert"
                expert_steps += 1

            transition = task.step(executed)
            env_steps += 1
            next_observation = _transition_observation(transition)
            if next_observation is None:
                next_observation = _observe_task(task, None)
            next_privileged = _privileged_state(task)
            tracker.update(next_privileged, transition)
            environment_done = _transition_terminal(transition)
            target_success = tracker.target_completed(target_id)
            target_failure = tracker.target_failed(target_id)
            prefix_valid = not tracker.expert_prefix_failed(target_id)
            completed_prefix_stages = {
                stage.id
                for stage in tracker.stages[:target_index]
            }
            prefix_skipped_boundary = (
                not handed_off
                and len(
                    completed_prefix_stages.intersection(
                        tracker.newly_completed_stages
                    )
                ) > 1
            )
            record = StageStepLog(
                task_id=str(getattr(task, "task_id", "unknown")),
                episode_seed=int(seed),
                target_stage=target_id,
                environment_step=env_steps,
                model_inference_index=inference_index,
                observation_id=env_steps - 1,
                observation_timestamp=_transition_timestamp(transition),
                privileged_stage_state=next_privileged,
                proposed_model_action=proposed,
                executed_action=executed,
                action_source=action_source,
                expert_prefix_valid=prefix_valid,
                target_success=target_success,
                target_failure=target_failure,
                model_control_steps=model_steps,
                environment_done=environment_done,
                stage=tracker.current_stage,
            )
            if record_logs:
                logs.append(record)
            _emit_log(logger, record)
            observation = next_observation

            # Success has priority over terminal/failure, matching the main
            # rollout evaluator's final-state ordering.
            if handed_off and target_success:
                success_step = model_steps
                return _result(
                    task, target_id, seed, SUCCESS, True, success_step, expert_steps,
                    model_steps, model_inference_calls, env_steps, handoff_observation_id,
                    None, logs,
                )
            if handed_off and target_failure:
                return _result(
                    task, target_id, seed, FAILURE, False, success_step, expert_steps,
                    model_steps, model_inference_calls, env_steps, handoff_observation_id,
                    "target_failure_predicate", logs,
                )
            if not handed_off and target_success:
                return _result(
                    task, target_id, seed, INVALID_PREFIX, False, None, expert_steps,
                    model_steps, model_inference_calls, env_steps, handoff_observation_id,
                    "expert_completed_target_during_prefix", logs,
                )
            if not handed_off and prefix_skipped_boundary:
                return _result(
                    task, target_id, seed, INVALID_PREFIX, False, None, expert_steps,
                    model_steps, model_inference_calls, env_steps, handoff_observation_id,
                    "expert_skipped_stage_boundary", logs,
                )
            if not handed_off and target_failure:
                return _result(
                    task, target_id, seed, INVALID_PREFIX, False, None, expert_steps,
                    model_steps, model_inference_calls, env_steps, handoff_observation_id,
                    "expert_target_failure_during_prefix", logs,
                )
            if not handed_off and not prefix_valid:
                return _result(
                    task, target_id, seed, INVALID_PREFIX, False, None, expert_steps,
                    model_steps, model_inference_calls, env_steps, handoff_observation_id,
                    "expert_prefix_failed", logs,
                )
            if environment_done:
                return _result(
                    task, target_id, seed,
                    FAILURE if handed_off else INVALID_PREFIX,
                    False,
                    success_step,
                    expert_steps,
                    model_steps,
                    model_inference_calls,
                    env_steps,
                    handoff_observation_id,
                    "environment_done" if handed_off else "expert_prefix_environment_done",
                    logs,
                )
    finally:
        _close_task(task)


def _task_stages(task: Any) -> tuple[StageSpec, ...]:
    value = getattr(task, "stage_specs", None)
    if value is None:
        value = getattr(task, "stages", None)
    if value is None:
        raise TypeError("stage task must expose stage_specs")
    stages = tuple(value)
    if not all(isinstance(stage, StageSpec) for stage in stages):
        raise TypeError("stage_specs must contain StageSpec values")
    return stages


def _task_instruction(task: Any) -> str:
    value = getattr(task, "instruction", None)
    if value is None:
        getter = getattr(task, "get_instruction", None)
        value = getter() if callable(getter) else None
    if not isinstance(value, str):
        raise TypeError("stage task must expose a string instruction")
    return value


def _reset_task(task: Any, seed: int) -> Any:
    reset = getattr(task, "reset", None)
    if not callable(reset):
        raise TypeError("stage task must expose reset(seed)")
    try:
        parameters = inspect.signature(reset).parameters
    except (TypeError, ValueError):
        parameters = {}
    if "seed" in parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    ):
        return reset(seed=seed)
    return reset(seed)


def _close_task(task: Any) -> None:
    close = getattr(task, "close", None)
    if callable(close):
        close()


def _observe_task(task: Any, fallback: Any) -> Any:
    observe = getattr(task, "observe", None)
    if not callable(observe):
        observe = getattr(task, "get_observation", None)
    if callable(observe):
        return observe()
    if fallback is None:
        raise TypeError("stage task must expose observe() when reset returns no observation")
    return fallback


def _privileged_state(task: Any) -> Any:
    getter = getattr(task, "get_privileged_state", None)
    if callable(getter):
        return getter()
    value = getattr(task, "privileged_state", None)
    if value is None:
        raise TypeError("stage task must expose evaluator-private privileged state")
    return value


def _transition_observation(transition: Any) -> Any:
    if isinstance(transition, tuple) and transition:
        return transition[0]
    if isinstance(transition, Mapping):
        value = transition.get("observation")
        if value is not None:
            return value
    return getattr(transition, "observation", None)


def _transition_terminal(transition: Any) -> bool:
    if isinstance(transition, tuple):
        if len(transition) == 4:
            return bool(transition[2])
        if len(transition) == 5:
            return bool(transition[2] or transition[3])
        return False
    if isinstance(transition, Mapping):
        if "terminal" in transition:
            return bool(transition["terminal"])
        if "done" in transition:
            return bool(transition["done"])
        return bool(transition.get("terminated", False) or transition.get("truncated", False))
    terminal = getattr(transition, "terminal", None)
    if terminal is not None:
        return bool(terminal)
    return bool(
        getattr(transition, "terminated", False)
        or getattr(transition, "truncated", False)
        or getattr(transition, "done", False)
    )


def _transition_timestamp(transition: Any) -> float | None:
    value = (
        transition.get("timestamp")
        if isinstance(transition, Mapping)
        else getattr(transition, "timestamp", None)
    )
    if value is None and isinstance(transition, Mapping):
        value = transition.get("observation_timestamp")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _action_rows(value: Any) -> list[Any]:
    array = np.asarray(value)
    if array.ndim == 0:
        return [value]
    if array.ndim == 1:
        return [value]
    if array.ndim == 2:
        if array.shape[0] == 0:
            raise ValueError("model returned an empty action chunk")
        return [row.copy() for row in array]
    raise ValueError(f"model action must be a vector or 2-D chunk, got shape {array.shape}")


def _emit_log(logger: Any, record: StageStepLog) -> None:
    if logger is None:
        return
    payload = record.to_dict()
    if callable(logger):
        logger(payload)
        return
    callback = getattr(logger, "record", None)
    if not callable(callback):
        raise TypeError("stage logger must be callable or expose record(payload)")
    callback(payload)


def _result(
    task: Any,
    target_id: str,
    seed: int,
    status: str,
    success: bool,
    success_step: int | None,
    expert_steps: int,
    model_steps: int,
    model_calls: int,
    env_steps: int,
    handoff_observation_id: int | None,
    failure_reason: str | None,
    logs: Sequence[StageStepLog],
) -> StageEvaluationResult:
    return StageEvaluationResult(
        task_id=str(getattr(task, "task_id", "unknown")),
        target_stage=target_id,
        seed=int(seed),
        status=status,
        success=bool(success),
        success_step=success_step,
        expert_prefix_steps=int(expert_steps),
        model_control_steps=int(model_steps),
        total_model_inference_calls=int(model_calls),
        environment_steps=int(env_steps),
        handoff_observation_id=handoff_observation_id,
        failure_reason=failure_reason,
        logs=tuple(logs),
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


__all__ = [
    "FAILURE",
    "FunctionalStageTask",
    "ExpertController",
    "INVALID_PREFIX",
    "StageEvaluationResult",
    "StageEvaluationTask",
    "ModelAdapter",
    "StageStatistics",
    "StageStatus",
    "StageSpec",
    "StageStepLog",
    "StageTask",
    "StageTracker",
    "SUCCESS",
    "aggregate_stage_results",
    "evaluate_stage",
    "prefix_validity",
    "macro_stage_sr",
    "stage_success_rate",
    "stage_sr",
    "wilson_interval",
]
