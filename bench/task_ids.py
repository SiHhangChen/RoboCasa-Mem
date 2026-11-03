"""Canonical public task IDs, with historical spellings accepted on input."""

TASK_IDS = (
    "WX-01", "WA-01", "WA-05", "WR-01", "WR-08",
    "TS-02", "TS-03", "TM-01", "TM-02",
)
_BY_KEY = {value.replace("-", "").casefold(): value for value in TASK_IDS}


def normalize_task_id(value: str) -> str:
    """Normalize builtin IDs while leaving third-party plugin IDs unchanged."""
    key = "".join(ch for ch in value.casefold() if ch.isalnum())
    return _BY_KEY.get(key, value)


__all__ = ["TASK_IDS", "normalize_task_id"]
