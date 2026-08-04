"""Expert-only stage orchestration over task-local physical executors."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from typing import Protocol

from bench.execution.contracts import ExpertPlan, NavigateStage, OperateStage, StageKind, StageResult


class NavigationExecutor(Protocol):
    def navigate_to(self, station_name: str, **kwargs: Any) -> Any: ...


class ManipulationExecutor(Protocol):
    def execute(self, operation: str, target: str, **kwargs: Any) -> Any: ...


@dataclass(frozen=True, slots=True)
class ExpertRunResult:
    task_id: str
    success: bool
    stages: tuple[StageResult, ...]
    details: dict[str, Any] = field(default_factory=dict)


class StageRunner:
    """Run an ExpertPlan and short-circuit after the first failed stage."""

    def __init__(self, *, navigation: NavigationExecutor, manipulation: ManipulationExecutor) -> None:
        self.navigation = navigation
        self.manipulation = manipulation

    def run(
        self,
        plan: ExpertPlan,
        *,
        on_stage_end: Callable[[int, Any, StageResult], None] | None = None,
    ) -> ExpertRunResult:
        results: list[StageResult] = []
        for index, stage in enumerate(plan.stages):
            try:
                if isinstance(stage, NavigateStage):
                    value = self._run_navigation(stage)
                    kind = StageKind.NAVIGATE
                elif isinstance(stage, OperateStage):
                    value = self._run_operation(stage)
                    kind = StageKind.OPERATE
                else:
                    raise TypeError(f"unsupported expert stage: {type(stage).__name__}")
                result = _stage_result(index, kind, value)
            except Exception as exc:
                result = StageResult(index=index, kind=stage.kind, success=False, failure_code="stage_error", details={"error": f"{type(exc).__name__}: {exc}"})
            results.append(result)
            if on_stage_end is not None:
                on_stage_end(index, stage, result)
            if not result.success:
                break
        return ExpertRunResult(plan.task_id, bool(results) and all(item.success for item in results) and len(results) == len(plan.stages), tuple(results))

    def _run_navigation(self, stage: NavigateStage) -> Any:
        target = stage.target
        metadata = {**dict(target.metadata), **dict(stage.metadata)}
        if stage.skip_if_reachable:
            probe = getattr(
                self.manipulation,
                "current_base_entry_reachability_summary",
                None,
            )
            if callable(probe):
                reachability = probe(
                    target.station_id,
                    purpose=target.purpose or "navigate",
                    holding=target.holding,
                    **metadata,
                )
                if bool(reachability.get("pass", False)):
                    return {
                        **dict(reachability),
                        "success": True,
                        "skipped": True,
                        "skip_reason": reachability.get(
                            "reason", "current_base_entry_stage_reachable"
                        ),
                    }
                if not stage.fallback_if_unreachable:
                    return {
                        **dict(reachability),
                        "success": False,
                        "skipped": False,
                        "failure_code": reachability.get(
                            "reason", "current_base_entry_stage_not_reachable"
                        ),
                    }
        value = self.navigation.navigate_to(
            target.station_id,
            purpose=target.purpose,
            holding=target.holding,
            **metadata,
        )
        self._sync_manipulation_after_navigation()
        return value

    def _run_operation(self, stage: OperateStage) -> Any:
        metadata = dict(stage.metadata)
        if stage.operation is not None:
            return self.manipulation.execute(
                stage.operation,
                stage.target or "",
                **metadata,
            )
        if not stage.segments:
            raise ValueError("operate stage needs an operation or motion segments")
        execute_segments = getattr(self.manipulation, "execute_segments", None)
        if callable(execute_segments):
            return execute_segments(stage.segments, **metadata)
        return self.manipulation.execute("motion_segments", "", segments=stage.segments, **metadata)

    def _sync_manipulation_after_navigation(self) -> None:
        sync_obs = getattr(self.manipulation, "_sync_obs", None)
        driver = getattr(self.navigation, "_driver", None)
        observation = getattr(driver, "_obs", None)
        if callable(sync_obs) and observation is not None:
            setattr(self.manipulation, "_pending_navigation_obs", observation)
            sync_obs(observation)
            return
        refresh = getattr(self.manipulation, "refresh_obs", None)
        if callable(refresh):
            refresh()


def _stage_result(index: int, kind: StageKind, value: Any) -> StageResult:
    if isinstance(value, StageResult):
        return StageResult(index=index, kind=kind, success=value.success, failure_code=value.failure_code, details=dict(value.details))
    if isinstance(value, bool):
        return StageResult(index=index, kind=kind, success=value)
    if value is None:
        return StageResult(index=index, kind=kind, success=True)
    if isinstance(value, dict):
        success = bool(value.get("success", value.get("pass", True)))
        failure = value.get("failure_code", value.get("reason"))
        details = dict(value)
    else:
        success = bool(getattr(value, "success", True))
        failure = getattr(value, "failure_code", None) or getattr(value, "error", None)
        summary = getattr(value, "summary", None)
        details = dict(summary or {}) if isinstance(summary, dict) else {}
    return StageResult(
        index=index,
        kind=kind,
        success=success,
        failure_code=None if success or failure is None else str(failure),
        details=details,
    )


__all__ = ["ExpertRunResult", "NavigationExecutor", "ManipulationExecutor", "StageRunner"]
