"""WA-03 权威语义状态、事件和唯一成功谓词。"""

from __future__ import annotations

from dataclasses import dataclass

from bench.tasks.wa03.domain.contracts import COFFEE_IDS


@dataclass
class SemanticState:
    """集中保存无法仅由当前物理画面推导的任务历史状态。"""

    target_sugared_coffee_id: str
    sugar_consumed: bool = False
    sugared_coffee_id: str | None = None
    unsugared_coffee_ids: tuple[str, ...] = ()
    all_cups_ready_before_selection: bool = False
    selected_sugared_coffee_id: str | None = None

    def __post_init__(self) -> None:
        """在状态创建时立即校验目标杯子标识。"""

        self.target_sugared_coffee_id = _coffee_id(self.target_sugared_coffee_id)


@dataclass(frozen=True)
class SuccessReport:
    """保存四项权威成功条件的布尔结果。"""

    sugar_consumed: bool
    sugar_assignment_correct: bool
    all_cups_ready_before_selection: bool
    correct_sugared_selection: bool

    @property
    def success(self) -> bool:
        """仅当四项任务条件全部成立时返回成功。"""

        return bool(
            self.sugar_consumed
            and self.sugar_assignment_correct
            and self.all_cups_ready_before_selection
            and self.correct_sugared_selection
        )

    def checks(self) -> dict[str, bool]:
        """输出便于写入 diagnostics 的成功条件字典。"""

        return {
            "sugar_consumed": self.sugar_consumed,
            "sugar_assignment_correct": self.sugar_assignment_correct,
            "all_cups_ready_before_selection": self.all_cups_ready_before_selection,
            "correct_sugared_selection": self.correct_sugared_selection,
        }


def set_target(state: SemanticState, coffee_id: str) -> None:
    """更新本 episode 指定的加糖目标杯。"""

    state.target_sugared_coffee_id = _coffee_id(coffee_id)


def record_sugar_delivery(state: SemanticState, coffee_id: str) -> bool:
    """首次记录糖被投递到哪杯咖啡，并确定另外两杯未加糖。"""

    coffee_id = _coffee_id(coffee_id)
    if state.sugar_consumed:
        return False
    state.sugar_consumed = True
    state.sugared_coffee_id = coffee_id
    state.unsugared_coffee_ids = tuple(value for value in COFFEE_IDS if value != coffee_id)
    return True


def latch_all_cups_ready(state: SemanticState, *, currently_ready: bool) -> bool:
    """在最终抓杯前锁存三杯曾正确就位这一历史事实。"""

    if currently_ready:
        state.all_cups_ready_before_selection = True
    return state.all_cups_ready_before_selection


def record_selection(state: SemanticState, coffee_id: str) -> None:
    """记录机器人最终选择并抓起的咖啡杯。"""

    state.selected_sugared_coffee_id = _coffee_id(coffee_id)


def evaluate_success(state: SemanticState) -> SuccessReport:
    """从统一语义状态计算唯一的最终成功报告。"""

    expected_unsugared = tuple(
        value for value in COFFEE_IDS if value != state.target_sugared_coffee_id
    )
    assignment_correct = bool(
        state.sugared_coffee_id == state.target_sugared_coffee_id
        and tuple(state.unsugared_coffee_ids) == expected_unsugared
    )
    selection_correct = bool(
        state.selected_sugared_coffee_id is not None
        and state.selected_sugared_coffee_id == state.sugared_coffee_id
    )
    return SuccessReport(
        sugar_consumed=bool(state.sugar_consumed),
        sugar_assignment_correct=assignment_correct,
        all_cups_ready_before_selection=bool(state.all_cups_ready_before_selection),
        correct_sugared_selection=selection_correct,
    )


def _coffee_id(value: str) -> str:
    """内部统一校验杯子标识，避免无效状态进入语义层。"""

    coffee_id = str(value)
    if coffee_id not in COFFEE_IDS:
        raise ValueError(f"invalid WA-03 coffee id: {coffee_id!r}")
    return coffee_id


__all__ = [
    "SemanticState",
    "SuccessReport",
    "evaluate_success",
    "latch_all_cups_ready",
    "record_selection",
    "record_sugar_delivery",
    "set_target",
]
