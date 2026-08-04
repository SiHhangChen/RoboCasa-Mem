"""Shared task-option lookup and conversion helpers.

Task bundles accept a loosely typed mapping of per-task overrides.  These
helpers centralize the common snake_case / kebab-case lookup convention while
keeping the historically distinct conversion policies explicit in their names.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from math import isfinite
from pathlib import Path
from typing import Any, TypeVar

import numpy as np


_TRUE_VALUES = frozenset({"1", "true", "yes", "y", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "n", "off"})
_DISABLED_VALUE_STRINGS = frozenset({"", "none", "off"})
T = TypeVar("T")


def option(options: Mapping[str, Any], name: str, default: Any = None) -> Any:
    """Read ``name`` or its kebab-case spelling, preferring ``name`` itself."""
    if name in options:
        return options[name]
    return options.get(name.replace("_", "-"), default)


def raw_str_option(options: Mapping[str, Any], name: str, default: Any = "") -> str:
    """Return the option stringification, preserving ``None`` as ``"None"``."""
    return str(option(options, name, default))


def empty_str_option(options: Mapping[str, Any], name: str, default: Any = "") -> str:
    """Return the option stringification, mapping ``None`` to an empty string."""
    value = option(options, name, default)
    return "" if value is None else str(value)


def bool_option(options: Mapping[str, Any], name: str, default: bool = False) -> bool:
    """Parse standard boolean spellings and fall back to ``default`` if invalid."""
    value = option(options, name, default)
    if isinstance(value, bool):
        return value
    raw = str(value).strip().lower()
    if raw in _TRUE_VALUES:
        return True
    if raw in _FALSE_VALUES:
        return False
    return bool(default)


def true_only_bool_option(options: Mapping[str, Any], name: str, default: bool = False) -> bool:
    """Return true only for the accepted truthy spellings.

    This intentionally retains the established policy where an invalid value does
    not fall back to ``default``.
    """
    value = option(options, name, default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in _TRUE_VALUES


def int_option(options: Mapping[str, Any], name: str, default: Any = 0) -> int:
    """Read and directly convert an integer option."""
    return int(option(options, name, default))


def defaulting_int_option(options: Mapping[str, Any], name: str, default: Any = 0) -> int:
    """Convert an integer option, treating an explicit ``None`` as the default."""
    value = option(options, name, default)
    return int(default if value is None else value)


def optional_int_option(
    options: Mapping[str, Any],
    name: str,
    default: Any = None,
) -> int | None:
    """Convert an optional integer; only ``None`` and ``""`` mean absent."""
    value = option(options, name, default)
    if value is None or value == "":
        return None
    return int(value)


def is_disabled_option_value(value: Any) -> bool:
    """Return whether a value uses the common ``none/off`` disabled spelling."""
    return value is None or str(value).strip().lower() in _DISABLED_VALUE_STRINGS


def sentinel_optional_int_option(
    options: Mapping[str, Any],
    name: str,
    default: Any = None,
) -> int | None:
    """Convert an optional integer accepting ``""``, ``"none"``, and ``"off"``."""
    value = option(options, name, default)
    if is_disabled_option_value(value):
        return None
    return int(value)


def float_option(options: Mapping[str, Any], name: str, default: Any = 0.0) -> float:
    """Read and directly convert a float option."""
    return float(option(options, name, default))


def defaulting_float_option(options: Mapping[str, Any], name: str, default: Any = 0.0) -> float:
    """Convert a float option, treating an explicit ``None`` as the default."""
    value = option(options, name, default)
    return float(default if value is None else value)


def optional_float_option(
    options: Mapping[str, Any],
    name: str,
    default: Any = None,
) -> float | None:
    """Convert an optional float; only an actual ``None`` means absent."""
    value = option(options, name, default)
    return None if value is None else float(value)


def sentinel_optional_float_option(
    options: Mapping[str, Any],
    name: str,
    default: Any = None,
) -> float | None:
    """Convert an optional float accepting ``""``, ``"none"``, and ``"off"``."""
    value = option(options, name, default)
    if is_disabled_option_value(value):
        return None
    return float(value)


def path_from_option_value(value: Any) -> Path | None:
    """Convert a sentinel-aware path value to ``Path`` or ``None``."""
    if is_disabled_option_value(value):
        return None
    return Path(value)


def path_option(options: Mapping[str, Any], name: str, default: Any = None) -> Path | None:
    """Read a sentinel-aware optional path option."""
    return path_from_option_value(option(options, name, default))


def empty_path_option(
    options: Mapping[str, Any],
    name: str,
    default: Any = None,
) -> Path | None:
    """Read an optional path where only ``None`` and ``""`` mean absent."""
    value = option(options, name, default)
    if value in (None, ""):
        return None
    return Path(value)


def optional_float_list(value: Any) -> list[float] | None:
    """Flatten an optional numeric value into a JSON-friendly float list."""
    if value is None:
        return None
    return np.asarray(value, dtype=float).reshape(-1).astype(float).tolist()


def finite_float_pair_option(
    options: Mapping[str, Any],
    name: str,
    default: tuple[float, float],
) -> tuple[float, float]:
    """Parse a finite two-value option without discarding blank items."""
    value = option(options, name, default)
    if isinstance(value, str):
        parts = [item.strip() for item in value.replace(";", ",").split(",")]
    else:
        parts = list(value)
    if len(parts) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    result = (float(parts[0]), float(parts[1]))
    if not all(isfinite(item) for item in result):
        raise ValueError(f"{name} contains non-finite values")
    return result


def filtered_float_pair_option(
    options: Mapping[str, Any],
    name: str,
    default: tuple[float, float],
) -> tuple[float, float]:
    """Parse two floats after dropping blank entries from strings or sequences."""
    value = option(options, name, default)
    if isinstance(value, str):
        raw_items = value.replace(";", ",").split(",")
    else:
        raw_items = list(value)
    values = [float(item) for item in raw_items if str(item).strip()]
    if len(values) != 2:
        raise ValueError(f"{name} must contain exactly two numeric values")
    return values[0], values[1]


def float_pair_option(
    options: Mapping[str, Any],
    name: str,
    default: tuple[float, float],
) -> tuple[float, float]:
    """Parse two floats, dropping blank entries only when the input is a string."""
    value = option(options, name, default)
    if isinstance(value, str):
        items = [item.strip() for item in value.replace(";", ",").split(",")]
        values = [float(item) for item in items if item]
    else:
        values = [float(item) for item in value]
    if len(values) != 2:
        raise ValueError(f"{name} must contain exactly two floats")
    return float(values[0]), float(values[1])


def optional_float_triple_option(
    options: Mapping[str, Any],
    name: str,
    default: Any = None,
) -> tuple[float, float, float] | None:
    """Parse an optional three-float option where only ``None`` and ``""`` disable it."""
    value = option(options, name, default)
    if value is None or value == "":
        return None
    if isinstance(value, str):
        items = [item.strip() for item in value.replace(";", ",").split(",") if item.strip()]
    else:
        items = list(value)
    if len(items) != 3:
        raise ValueError(f"{name} must contain exactly three numeric values")
    return float(items[0]), float(items[1]), float(items[2])


def sentinel_optional_float_triple_option(
    options: Mapping[str, Any],
    name: str,
    default: Any = None,
) -> tuple[float, float, float] | None:
    """Parse a comma-separated optional triple accepting ``none/off`` sentinels."""
    value = option(options, name, default)
    if is_disabled_option_value(value):
        return None
    if isinstance(value, str):
        parts = [item.strip() for item in value.split(",") if item.strip()]
    else:
        parts = list(value)
    if len(parts) != 3:
        raise ValueError(f"{name} must be a comma-separated x,y,z triple")
    return float(parts[0]), float(parts[1]), float(parts[2])


def parse_finite_optional_float_pair(value: Any) -> tuple[float, float] | None:
    """Parse a finite pair from a raw optional value accepting ``none/off``."""
    if value is None:
        return None
    if isinstance(value, str):
        raw = value.strip()
        if raw.lower() in _DISABLED_VALUE_STRINGS:
            return None
        parts = [
            item.strip()
            for item in raw.replace(";", ",").split(",")
            if item.strip()
        ]
    else:
        parts = list(value)
    if len(parts) != 2:
        raise ValueError(f"xy option must contain exactly two values, got {value!r}")
    result = (float(parts[0]), float(parts[1]))
    if not all(isfinite(item) for item in result):
        raise ValueError(f"xy option contains non-finite values: {value!r}")
    return result


def first_option(
    options: Mapping[str, Any],
    *names: str,
    default: Any = None,
) -> Any:
    """Return the first present option, treating an empty string as present."""
    for name in names:
        value = option(options, name, None)
        if value is not None:
            return value
    return default


def first_nonempty_option(
    options: Mapping[str, Any],
    names: Iterable[str],
    default: Any = None,
) -> Any:
    """Return the first option that is neither ``None`` nor an empty string."""
    for name in names:
        value = option(options, name)
        if value is not None and value != "":
            return value
    return default


def first_present_option(
    options: Mapping[str, Any],
    kwargs: Mapping[str, Any],
    names: Iterable[str],
    default: Any = None,
) -> Any:
    """Find the first non-``None`` exact-name value in options, then kwargs."""
    for name in names:
        if name in options and options[name] is not None:
            return options[name]
        if name in kwargs and kwargs[name] is not None:
            return kwargs[name]
    return default


def optional_float_from_option_aliases(
    options: Mapping[str, Any],
    kwargs: Mapping[str, Any],
    names: Iterable[str],
) -> float | None:
    """Read the first alias across two mappings and convert a sentinel-aware float."""
    value = first_present_option(options, kwargs, names, None)
    if is_disabled_option_value(value):
        return None
    return float(value)


def set_optional_metadata(metadata: dict[str, Any], key: str, value: Any) -> None:
    """Set metadata unless its value is ``None`` or exactly an empty string."""
    if value is None or value == "":
        return
    metadata[key] = value


def set_nonblank_metadata(metadata: dict[str, Any], key: str, value: Any) -> None:
    """Set metadata unless its value is ``None`` or a blank string."""
    if value is None:
        return
    if isinstance(value, str) and not value.strip():
        return
    metadata[key] = value


def runtime_option(
    options: Mapping[str, Any],
    kwargs: Mapping[str, Any],
    name: str,
    *,
    default: T,
    cli_default: T,
    convert: Callable[[Any], T],
) -> T:
    """Resolve a runtime scalar while preserving the CLI-default rule."""
    value = option(options, name)
    if value is not None:
        return convert(value)
    value = kwargs.get(name, default)
    if convert(value) == convert(cli_default):
        return convert(default)
    return convert(value)


def runtime_float_option(
    options: Mapping[str, Any],
    kwargs: Mapping[str, Any],
    name: str,
    *,
    default: float,
    cli_default: float,
) -> float:
    """Resolve a runtime float option."""
    return runtime_option(
        options,
        kwargs,
        name,
        default=default,
        cli_default=cli_default,
        convert=float,
    )


def runtime_int_option(
    options: Mapping[str, Any],
    kwargs: Mapping[str, Any],
    name: str,
    *,
    default: int,
    cli_default: int,
) -> int:
    """Resolve a runtime integer option."""
    return runtime_option(
        options,
        kwargs,
        name,
        default=default,
        cli_default=cli_default,
        convert=int,
    )


__all__ = [
    "bool_option",
    "defaulting_float_option",
    "defaulting_int_option",
    "empty_path_option",
    "empty_str_option",
    "filtered_float_pair_option",
    "finite_float_pair_option",
    "first_nonempty_option",
    "first_option",
    "first_present_option",
    "float_option",
    "float_pair_option",
    "int_option",
    "is_disabled_option_value",
    "option",
    "optional_float_from_option_aliases",
    "optional_float_list",
    "optional_float_option",
    "optional_float_triple_option",
    "optional_int_option",
    "parse_finite_optional_float_pair",
    "path_from_option_value",
    "path_option",
    "raw_str_option",
    "runtime_float_option",
    "runtime_int_option",
    "runtime_option",
    "sentinel_optional_float_option",
    "sentinel_optional_float_triple_option",
    "sentinel_optional_int_option",
    "set_nonblank_metadata",
    "set_optional_metadata",
    "true_only_bool_option",
]
