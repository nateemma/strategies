"""032 T013: the train-only driver refuses a concurrent chain and resumes."""

from __future__ import annotations

import pytest

from research.candidate_training import train as T


PS_IDLE = ["  2661 /x/python3 /x/freqtrade trade --dry-run -c cfg --strategy MomentumRegimeBasket15mFast",
           "   500 -zsh"]


def test_other_chains_ignores_the_dry_run_bot_and_itself():
    assert T.other_chains(PS_IDLE, own_pid=999) == []
    own = ["   999 python -m research.candidate_training.train --model ARO"]
    assert T.other_chains(own, own_pid=999) == []


@pytest.mark.parametrize("cmd", [
    "zsh user_data/strategies/scripts/test_strat.sh -c cfg -f NNNC X",
    "python -m evaluation.run_folds --plan p.json",
    "python -m regime.run_arm --strategy X",
    "/x/freqtrade backtesting -c cfg",
    "python -m research.candidate_training.train --model GBR",
])
def test_other_chains_flags_a_running_chain(cmd):
    assert T.other_chains([f"  123 {cmd}"], own_pid=999) == [cmd]


class FakeRF:
    def __init__(self, have: set):
        self.have, self.trained, self.generated = set(have), [], []

    def generate_fold(self, f, base, family, params, seed=1):
        self.generated.append(f["fold_id"])

    def fold_has_model(self, f, base, config, seed):
        return f["fold_id"] in self.have

    def train_fold(self, f, family, seed, config):
        self.trained.append(f["fold_id"])
        self.have.add(f["fold_id"])


def test_refuses_when_another_chain_runs():
    with pytest.raises(SystemExit, match="REFUSED"):
        T.run("ARO", {"f00"}, rf=FakeRF(set()), ps=lambda: ["  123 python -m evaluation.run_folds"],
              own_pid=999)


def test_resume_skips_trained_folds(monkeypatch, tmp_path):
    monkeypatch.setattr(T, "_location", lambda *a: type("L", (), {"path": str(tmp_path / "m.keras")}))
    rf = FakeRF({"f00"})
    out = T.run("ARO", {"f00", "f01"}, rf=rf, ps=lambda: PS_IDLE, own_pid=999)
    assert rf.generated == ["f00", "f01"]
    assert rf.trained == ["f01"]
    assert [o["trained"] for o in out] == [False, True]


def test_stops_when_training_leaves_no_model():
    class Broken(FakeRF):
        def train_fold(self, f, family, seed, config):
            self.trained.append(f["fold_id"])

    with pytest.raises(SystemExit, match="NO MODEL"):
        T.run("ARO", {"f00"}, rf=Broken(set()), ps=lambda: PS_IDLE, own_pid=999)
