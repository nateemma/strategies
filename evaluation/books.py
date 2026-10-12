"""Separate books: several single-family arms run side by side as independent sub-portfolios
(026 addendum 2).

Each book is its own walk-forward arm (own backtest, own one-position-per-pair book), so no
family can re-sequence another's entries. The combined result is the per-fold SUM of the books'
wallet returns at the config's per-trade stake, with the trades pooled as a union -- the same
shape `pool()` returns, so `halves.split` gives the usual paired late verdict against a bar.

The sum is only honest if the books fit in the one wallet the comparison assumes. This reports
how often they would not: entries made while the books together already held `max_open_trades`
positions (a shared wallet would have refused them), and entries on a pair another book held
(fine on two sub-accounts, impossible in one freqtrade bot). PROVISIONAL under quality gate 2.

    python -m evaluation.books --books st_bgo_mx,st_book_rev --bar st_bgo_mx \\
        --out specs/026-reversion-regime-gate/books.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


STRAT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STRAT))

FOLDS = STRAT / "evaluation" / "folds"
LABEL = "PROVISIONAL -- not a finding under quality gate 2 (no distant era)"


def combine(pooled: dict[str, dict], stake_scale: float = 1.0) -> dict:
    """Per-fold return = sum over books; trades = union tagged with their book.

    `stake_scale` multiplies every book's stake (fold returns and trade profit_abs), e.g. 0.5 =
    two books in ONE wallet at half the per-trade stake with twice the slots -- same capital as
    a single-book arm. Linear in stake: fixed-stake books, so no compounding is lost."""
    tags = list(pooled)
    folds = [f["fold_id"] for f in pooled[tags[0]]["per_fold"]]
    for t in tags[1:]:
        if [f["fold_id"] for f in pooled[t]["per_fold"]] != folds:
            raise ValueError(f"{t}: fold set differs from {tags[0]}")
    per_fold = []
    for i, fid in enumerate(folds):
        rows = [pooled[t]["per_fold"][i] for t in tags]
        rets = [r["return_pct"] for r in rows]
        per_fold.append({
            "fold_id": fid,
            "n_trades": sum(r["n_trades"] for r in rows),
            "return_pct": (None if all(x is None for x in rets)
                           else stake_scale * sum(x or 0.0 for x in rets)),
            "test_days": rows[0]["test_days"],
            "status": "evaluated",
        })
    trades = [{**tr, "book": t, "profit_abs": tr["profit_abs"] * stake_scale}
              for t in tags for tr in pooled[t]["trades"]]
    missing = sorted({m for t in tags for m in pooled[t].get("folds_missing", [])})
    return {
        **pooled[tags[0]],
        "run_id": "+".join(pooled[t]["run_id"] for t in tags),
        "family": "books",
        "per_fold": per_fold,
        "trades": trades,
        "n_trades_pooled": len(trades),
        "folds_complete": len(per_fold),
        "folds_missing": missing,
        "status": "complete" if not missing else "partial",
    }


def capacity(trades: list[dict], max_open: int) -> dict:
    """Entries a single shared wallet / single bot could not have made."""
    df = pd.DataFrame(trades)
    df["o"] = pd.to_datetime(df["open_date"], utc=True)
    df["c"] = pd.to_datetime(df["close_date"], utc=True)
    over, same_pair, peak = 0, 0, 0
    for _, g in df.groupby("fold_id"):
        g = g.sort_values("o")
        o, c = g["o"].to_numpy(), g["c"].to_numpy()
        pair, book = g["pair"].to_numpy(), g["book"].to_numpy()
        for i in range(len(g)):
            held = (o < o[i]) & (c > o[i])
            n = int(held.sum())
            peak = max(peak, n + 1)
            if n >= max_open:
                over += 1
            if (held & (pair == pair[i]) & (book != book[i])).any():
                same_pair += 1
    return {"entries": len(df), "over_max_open": over, "same_pair_other_book": same_pair,
            "peak_open": peak, "max_open_trades": max_open}


def _pooled(tag: str) -> dict:
    from evaluation.pool import pool

    return pool(json.loads((FOLDS / f"plan_{tag}.json").read_text()))


def _halves(pooled: dict) -> dict:
    rets = {f["fold_id"]: f["return_pct"] or 0.0 for f in pooled["per_fold"]}
    early = sum(v for k, v in rets.items() if int(k[1:]) < 8)
    return {"total": early + sum(v for k, v in rets.items() if int(k[1:]) >= 8),
            "early": early, "late": sum(v for k, v in rets.items() if int(k[1:]) >= 8),
            "positive": sum(v > 0 for v in rets.values())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--books", required=True, help="comma-separated arm tags")
    ap.add_argument("--bar", required=True)
    ap.add_argument("--stake-scale", type=float, default=1.0,
                    help="per-trade stake multiplier for every book (0.5 = same capital, 2x slots)")
    ap.add_argument("--config", default=str(STRAT / "config" / "config_wf.json"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from evaluation.fingerprint import load_config
    from evaluation.halves import split

    tags = [t for t in a.books.split(",") if t]
    books = combine({t: _pooled(t) for t in tags}, a.stake_scale)
    bar = _pooled(a.bar)
    late = split(books, bar)["late"]
    whole = split(books, bar, split_after="f14")["early"]
    slots = int(round(int(load_config(a.config)["max_open_trades"]) / a.stake_scale))
    cap = capacity(books["trades"], slots)
    h = _halves(books)
    bs, ws = late, whole
    lines = [
        f"## Books: {' + '.join(f'`{t}`' for t in tags)} (stake x{a.stake_scale:g}, {slots} slots)",
        "",
        f"*{LABEL}*",
        "",
        (f"total {h['total']:+.2f} pp, positive folds {h['positive']}/15, "
         f"early {h['early']:+.2f}, late {h['late']:+.2f}"),
        "",
        (f"late (f08-f14) vs `{a.bar}`: **{bs.get('verdict')}**, "
         f"{json.dumps({k: bs.get(k) for k in ('delta_pp', 'ci_pp_yr', 'p')}, default=str)}"),
        "",
        (f"whole span vs `{a.bar}`: **{ws.get('verdict')}**, "
         f"{json.dumps({k: ws.get(k) for k in ('delta_pp', 'ci_pp_yr', 'p')}, default=str)}"),
        "",
        f"capacity: {json.dumps(cap)}",
        "",
    ]
    out = Path(a.out)
    with out.open("a") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
