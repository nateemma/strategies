"""T022 — every LabelMethod is mapped exactly once (023 FR-004, data-model LabelTriggerMapping)."""
from __future__ import annotations

import json

import pytest

from Framework.TrainingSignals import LabelMethod
from research.outcomes import triggers as T
from research.outcomes.causality import VERDICT_FILE
from SimpleTrigger import label_map as LM


def test_exactly_one_row_per_member():
    assert set(LM.MAPPING) == set(LabelMethod)
    assert len(LM.MAPPING) == len(LabelMethod) == 29


def test_unsupported_rows_have_reason_and_no_trigger():
    for m, row in LM.MAPPING.items():
        assert row.label_method is m
        if row.status == LM.UNSUPPORTED:
            assert row.trigger is None and row.reason.strip(), m
            with pytest.raises(ValueError):
                LM.trigger_for(m)
        else:
            assert row.trigger, m


def test_triggers_are_registered_and_causal():
    T.ensure_builtins()
    verdicts = json.loads(VERDICT_FILE.read_text())["triggers"]
    for m, row in LM.MAPPING.items():
        if row.trigger:
            assert row.trigger in T.names(), (m, row.trigger)
            assert verdicts[row.trigger]["verdict"] == "PASS", (m, row.trigger)


def test_empty_rows_are_the_eight_pure_outcome_labels():
    empty = {m.name for m, r in LM.MAPPING.items() if r.status == LM.EMPTY}
    assert empty == {"forward_mae", "triple_barrier", "quantile_future", "mfe_mae_ratio",
                     "risk_adj_future", "multi_horizon_vote", "optimal_signals", "local_extrema"}
    assert all(LM.MAPPING[LabelMethod[n]].trigger == "empty" for n in empty)
