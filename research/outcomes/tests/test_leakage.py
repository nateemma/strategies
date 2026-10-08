# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
"""T013 (specs/021-label-exit-coordination) — the two-layer leakage guard (FR-009),
written BEFORE `table.py` (T014) produces any `__outcome_*` column.

Layer 1 (reserved prefix) is pinned against `OUTCOME_COLUMN_NAMES` -- defined in
`leakage.py`, not `table.py`, precisely so this file does not need the builder to
exist yet. Layer 2 (the raising assertion) is pinned against `build_label`, the
label-builder boundary.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from Framework.FeatureNormalizer import FeatureNormalizer  # noqa: E402
from research.outcomes.leakage import (  # noqa: E402
    OUTCOME_COLUMN_NAMES,
    OUTCOME_COLUMN_POSITIVE,
    OUTCOME_COLUMN_PROFIT,
    OUTCOME_PREFIX,
    LeakageError,
    assert_no_outcome_columns,
    build_label,
    outcome_columns,
)


# ---------------------------------------------------------------------------------------
# Layer 1: the reserved prefix cannot be selected by FeatureNormalizer.include_list.
# ---------------------------------------------------------------------------------------


def test_every_outcome_column_name_carries_the_reserved_prefix():
    assert len(OUTCOME_COLUMN_NAMES) == 7, "data-model.md's OutcomeRecord lists 7 fields"
    assert all(name.startswith(OUTCOME_PREFIX) for name in OUTCOME_COLUMN_NAMES)


def test_outcome_column_names_are_disjoint_from_feature_normalizer_include_list():
    """`FeatureNormalizer.include_list` is the fixed, hand-maintained list model input
    is built from (contracts/outcome-table.md). None of the seven outcome columns may
    appear in it -- if one did, a model could select it by accident regardless of the
    prefix convention."""
    include_list = set(FeatureNormalizer.include_list)
    assert OUTCOME_COLUMN_NAMES.isdisjoint(include_list)


def test_outcome_columns_helper_finds_only_prefixed_columns():
    df = pd.DataFrame({
        "close": [1.0],
        OUTCOME_COLUMN_PROFIT: [0.01],
        "rsi_scaled": [0.2],
    })
    assert outcome_columns(df) == [OUTCOME_COLUMN_PROFIT]


# ---------------------------------------------------------------------------------------
# Layer 2: the raising assertion at the label-builder boundary.
# ---------------------------------------------------------------------------------------


def test_assert_no_outcome_columns_passes_on_a_clean_frame():
    df = pd.DataFrame({"close": [1.0], "rsi_scaled": [0.2]})
    assert_no_outcome_columns(df, boundary="test")  # must not raise


def test_assert_no_outcome_columns_raises_on_a_contaminated_frame():
    df = pd.DataFrame({"close": [1.0], OUTCOME_COLUMN_PROFIT: [0.01]})
    with pytest.raises(LeakageError, match=OUTCOME_COLUMN_PROFIT):
        assert_no_outcome_columns(df, boundary="test boundary")


def test_build_label_forms_a_clean_label_from_outcome_positive():
    features = pd.DataFrame({"close": [1.0, 2.0], "rsi_scaled": [0.1, 0.2]})
    outcomes = pd.DataFrame({OUTCOME_COLUMN_POSITIVE: [True, False]})

    result = build_label(features, outcomes)

    assert list(result["label"]) == [True, False]
    assert outcome_columns(result) == []  # nothing crossed the boundary


def test_feeding_an_outcome_bearing_frame_into_the_label_path_raises():
    """T013's literal acceptance case: a `features` frame that (by mistake) still
    carries an outcome column must make the label builder raise, not silently train
    on it."""
    leaked_features = pd.DataFrame({
        "close": [1.0, 2.0],
        OUTCOME_COLUMN_PROFIT: [0.01, -0.02],  # the leak
    })
    outcomes = pd.DataFrame({OUTCOME_COLUMN_POSITIVE: [True, False]})

    with pytest.raises(LeakageError, match=OUTCOME_COLUMN_PROFIT):
        build_label(leaked_features, outcomes)


def test_build_label_raises_if_label_source_missing():
    features = pd.DataFrame({"close": [1.0]})
    outcomes = pd.DataFrame({"not_the_right_column": [True]})
    with pytest.raises(LeakageError):
        build_label(features, outcomes)
