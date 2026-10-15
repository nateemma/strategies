"""035 T009: the truncation harness passes every fixture input and fails the planted control."""

from __future__ import annotations

from research.new_inputs import causality as CA


def test_every_input_is_prefix_stable(synth):
    res = CA.run(synth, 12)
    assert res["cuts"] == 12 and res["checked"] > 50
    assert res["pass"], res["mismatches"][:5]


def test_planted_centred_control_fails(synth):
    with CA.planted_centred_taker_share():
        res = CA.run(synth, 6)
    assert not res["pass"]
    assert {m["column"] for m in res["mismatches"]} >= {"ni_of_tbg"}
