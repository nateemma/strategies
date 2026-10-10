"""Matched-count selection test (feature 023, research R13, FR-014).

THE QUESTION. Does a model's choice of trades beat a random choice of the SAME NUMBER of trades
from the population it could have chosen from? Feature 022 answered exactly this for its guard
constant (+5.2 SD, p < 1e-5) when the paired bootstrap could not: two arms that enter on different
bars share almost no trades (111 of ~8,500 there), so a trade-paired test has nothing to pair.
This test needs no shared trades. It was done by hand in 022; it is code here.

THE DRAW. Stratified by fold: for each fold, draw as many trades from the population's trades in
that fold as the selection has there, without replacement. Stratifying keeps each draw in the
same market regime as the selection -- an unstratified draw would credit the model for WHEN it
traded, not for WHICH trades it picked. The statistic is the summed pp return of the draw; the
null is its distribution over `n_draws` draws.

WHAT IT DOES NOT SAY. It ranks the selection against random subsets of the population's REALISED
trades. A subset of trades realised under one slot-occupancy pattern is not the trade list a
different entry rule would realise (max_open_trades interacts with entry timing), and the
selection need not be a subset of the population at all -- `overlap` reports how much it is.
Under quality gate 2 every result here is PROVISIONAL (no distant era).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "user_data/strategies"))

SEED = 0
N_DRAWS = 20000
PROVISIONAL = "PROVISIONAL -- not a finding under quality gate 2 (no distant era)"


def _pp(trades: list[dict]) -> np.ndarray:
    return np.array([t["profit_abs"] * t["scale_pp_per_abs"] for t in trades], dtype=float)


def matched_count(
    selected: list[dict],
    population: list[dict],
    n_draws: int = N_DRAWS,
    seed: int = SEED,
    folds: list[str] | None = None,
) -> dict:
    """Selection vs same-count random draws, stratified by `fold_id`."""
    by_fold_sel: dict[str, np.ndarray] = {}
    by_fold_pop: dict[str, np.ndarray] = {}
    for t in selected:
        by_fold_sel.setdefault(t["fold_id"], []).append(t)
    for t in population:
        by_fold_pop.setdefault(t["fold_id"], []).append(t)
    fold_ids = sorted(folds if folds is not None else by_fold_sel)
    rng = np.random.default_rng(seed)
    null = np.zeros(n_draws)
    observed = 0.0
    n_sel = 0
    for fid in fold_ids:
        sel = _pp(by_fold_sel.get(fid, []))
        pop = _pp(by_fold_pop.get(fid, []))
        k = len(sel)
        if k == 0:
            continue
        if k > len(pop):
            raise ValueError(
                f"fold {fid}: selection has {k} trades but the population only "
                f"{len(pop)}; a same-count draw without replacement is impossible"
            )
        observed += sel.sum()
        n_sel += k
        # argpartition of uniform keys == a uniform k-subset per draw, vectorised
        keys = rng.random((n_draws, len(pop)))
        idx = np.argpartition(keys, k - 1, axis=1)[:, :k]
        null += pop[idx].sum(axis=1)
    if n_sel == 0:
        raise ValueError("selection is empty")
    mu, sd = float(null.mean()), float(null.std(ddof=1))

    def key(t):
        return (t["fold_id"], t["pair"], t["open_date"])

    pop_keys = {key(t) for t in population}
    overlap = sum(1 for t in selected if key(t) in pop_keys)
    return {
        "n_selected": n_sel,
        "n_population": len(population),
        "folds": fold_ids,
        "observed_pp": float(observed),
        "null_mean_pp": mu,
        "null_sd_pp": sd,
        "z": float((observed - mu) / sd) if sd > 0 else float("nan"),
        "p_one_sided": float((np.sum(null >= observed) + 1) / (n_draws + 1)),
        "overlap": overlap,
        "overlap_fraction": overlap / n_sel,
        "n_draws": n_draws,
        "seed": seed,
        "label": PROVISIONAL,
    }


def matched_count_pooled(selected_pooled: dict, population_pooled: dict, **kw) -> dict:
    """Pooled-file form, plus the early/late halves (f00-f07 / f08-f14)."""
    out = {"pooled": matched_count(selected_pooled["trades"], population_pooled["trades"], **kw)}
    folds = sorted({t["fold_id"] for t in selected_pooled["trades"]})
    for name, ids in (
        ("early", [f for f in folds if f <= "f07"]),
        ("late", [f for f in folds if f > "f07"]),
    ):
        out[name] = (
            matched_count(selected_pooled["trades"], population_pooled["trades"], folds=ids, **kw)
            if ids
            else None
        )
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selected", required=True, help="pooled JSON of the selecting arm")
    ap.add_argument("--population", required=True, help="pooled JSON of the population arm")
    a = ap.parse_args(argv)
    res = matched_count_pooled(
        json.loads(Path(a.selected).read_text()), json.loads(Path(a.population).read_text())
    )
    print(PROVISIONAL)
    for name in ("pooled", "early", "late"):
        r = res[name]
        if r:
            print(
                f"  {name:<6} n={r['n_selected']:>5}  observed {r['observed_pp']:+9.3f}pp  "
                f"null {r['null_mean_pp']:+9.3f} ± {r['null_sd_pp']:.3f}  z={r['z']:+.2f}  "
                f"p={r['p_one_sided']:.5f}  overlap {r['overlap_fraction']:.1%}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
