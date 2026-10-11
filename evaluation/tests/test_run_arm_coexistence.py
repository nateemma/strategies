"""025 T006 (contract C3): rule A's refusals reach the run record via the sink JSONL."""

from __future__ import annotations

import json

from regime.run_arm import aggregate_coexistence


def test_jsonl_lines_are_summed_per_rule_and_family(tmp_path):
    lines = [{"rule": "pair_cap", "family": "reversion"}] * 2 + [
        {"rule": "pair_cap", "family": "momentum"}
    ]
    (tmp_path / "coexist_events.jsonl").write_text(
        "\n".join(json.dumps(x) for x in lines) + "\nnot json\n\n"
    )
    assert aggregate_coexistence(tmp_path) == {"pair_cap": {"momentum": 1, "reversion": 2}}


def test_missing_or_empty_sink_gives_empty(tmp_path):
    assert aggregate_coexistence(None) == {}
    assert aggregate_coexistence(tmp_path / "nope") == {}
    assert aggregate_coexistence(tmp_path) == {}
