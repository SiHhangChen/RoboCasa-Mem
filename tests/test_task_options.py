from __future__ import annotations

from pathlib import Path

import pytest

from bench.tasks._options import (
    bool_option,
    defaulting_int_option,
    empty_path_option,
    empty_str_option,
    filtered_float_pair_option,
    first_nonempty_option,
    first_option,
    first_present_option,
    float_pair_option,
    int_option,
    option,
    optional_float_list,
    optional_int_option,
    path_option,
    raw_str_option,
    runtime_int_option,
    sentinel_optional_int_option,
    set_nonblank_metadata,
    set_optional_metadata,
    true_only_bool_option,
)


def test_option_alias_lookup_preserves_explicit_none_and_precedence():
    assert option({"some-value": 3}, "some_value") == 3
    assert option({"some_value": None, "some-value": 3}, "some_value") is None
    assert option({}, "some_value", 5) == 5


def test_string_and_boolean_policy_variants_are_explicit():
    assert raw_str_option({"value": None}, "value") == "None"
    assert empty_str_option({"value": None}, "value") == ""
    assert bool_option({"value": "invalid"}, "value", True)
    assert not true_only_bool_option({"value": "invalid"}, "value", True)


def test_integer_optional_policies_preserve_legacy_sentinels():
    with pytest.raises(TypeError):
        int_option({"value": None}, "value", 4)
    assert defaulting_int_option({"value": None}, "value", 4) == 4
    with pytest.raises(ValueError):
        optional_int_option({"value": "off"}, "value")
    assert sentinel_optional_int_option({"value": "off"}, "value") is None


def test_shared_path_and_pair_parsers_keep_their_distinct_rules():
    assert path_option({"path": "off"}, "path") is None
    assert empty_path_option({"path": "off"}, "path") == Path("off")
    assert filtered_float_pair_option({"xy": ["", 1, 2]}, "xy", (0.0, 0.0)) == (1.0, 2.0)
    assert float_pair_option({"xy": "1,,2"}, "xy", (0.0, 0.0)) == (1.0, 2.0)


def test_alias_and_runtime_resolution_precedence():
    assert first_nonempty_option({"first": "", "second": 2}, ("first", "second")) == 2
    assert first_option({"first": ""}, "first", "second", default=3) == ""
    assert first_present_option({"second": 2}, {"first": 1}, ("first", "second")) == 1
    assert runtime_int_option({}, {"steps": 10}, "steps", default=4, cli_default=10) == 4
    assert runtime_int_option({"steps": 7}, {"steps": 10}, "steps", default=4, cli_default=10) == 7


def test_optional_metadata_and_float_list_policies():
    metadata: dict[str, object] = {}
    set_optional_metadata(metadata, "empty", "")
    set_nonblank_metadata(metadata, "blank", "  ")
    set_optional_metadata(metadata, "zero", 0)
    assert metadata == {"zero": 0}
    assert optional_float_list([[1, 2]]) == [1.0, 2.0]
