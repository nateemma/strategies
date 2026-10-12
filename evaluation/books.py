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

import numpy as np
import pandas as pd


STRAT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STRAT))

FOLDS = STRAT / "evaluation" / "folds"
LABEL = "PROVISIONAL -- not a finding under quality gate 2 (no distant era)"
# full-stake fills land at 899.0-899.9999 of 900 (amount precision rounding); below this a
# fill was cut by the liquidity cap
FULL_FILL_TOL = 0.002
# exchange minimum order in USDT (binance.us notional floor, approximate); a shared-wallet
# entry with less room than this is refused rather than shrunk
MIN_STAKE = 10.0


def resize(tr: dict, w: float, stake: float | None) -> tuple[float, float | None, bool]:
    """(profit multiplier, new stake, new stake is an upper bound) for one trade at weight w.

    BaseStrategy.custom_stake_amount fills min(proposed, candle quote volume / 10). A trade
    filled below the config stake was capped, so its cap IS its filled stake and a resized
    stake is min(w x stake, filled) -- NOT w x filled (026 addendum 5: ~75% of entries are
    capped). An uncapped trade's cap is only known to be >= stake: exact for w <= 1, an upper
    bound for w > 1. Without the filled stake (old archives) or config stake: linear in w."""
    s = tr.get("stake_amount")
    if stake is None or not s:
        return w, None, False
    capped = s < stake * (1 - FULL_FILL_TOL)
    new = min(w * stake, s) if capped else w * stake
    return new / s, new, (not capped and w > 1)


def combine(pooled: dict[str, dict], stake_scale: float = 1.0,
            weights: dict[str, float] | None = None, stake: float | None = None) -> dict:
    """Per-fold return = sum over books; trades = union tagged with their book.

    `stake_scale` multiplies every book's stake (fold returns and trade profit_abs), e.g. 0.5 =
    two books in ONE wallet at half the per-trade stake with twice the slots -- same capital as
    a single-book arm. Fixed-stake books, so no compounding is lost.
    `weights` ({book: w}) scales each book's stake on top; weights summing to 1 keep the
    capital of a single-book arm. With the config `stake`, each trade is resized through the
    liquidity cap (`resize`); without it, linearly."""
    tags = list(pooled)
    w = {t: stake_scale * (weights or {}).get(t, 1.0) for t in tags}
    trades, extra = [], {}
    for t in tags:
        for tr in pooled[t]["trades"]:
            m, new, upper = resize(tr, w[t], stake)
            trades.append({**tr, "book": t, "profit_abs": tr["profit_abs"] * m, "weight": w[t],
                           "stake": new, "stake_upper_bound": upper})
            # the fold return is w x the recorded return; add what the cap changes
            pp = (tr["profit_abs"] or 0.0) * (tr["scale_pp_per_abs"] or 0.0)
            extra[tr["fold_id"]] = extra.get(tr["fold_id"], 0.0) + (m - w[t]) * pp
    folds = [f["fold_id"] for f in pooled[tags[0]]["per_fold"]]
    for t in tags[1:]:
        if [f["fold_id"] for f in pooled[t]["per_fold"]] != folds:
            raise ValueError(f"{t}: fold set differs from {tags[0]}")
    per_fold = []
    for i, fid in enumerate(folds):
        rows = [pooled[t]["per_fold"][i] for t in tags]
        rets = [None if r["return_pct"] is None else w[t] * r["return_pct"]
                for t, r in zip(tags, rows, strict=True)]
        per_fold.append({
            "fold_id": fid,
            "n_trades": sum(r["n_trades"] for r in rows),
            "return_pct": (None if all(x is None for x in rets)
                           else sum(x or 0.0 for x in rets) + extra.get(fid, 0.0)),
            "test_days": rows[0]["test_days"],
            "status": "evaluated",
        })
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


def risk_weights(pooled: dict[str, dict], fold_ids: set[str]) -> dict[str, dict]:
    """Inverse-volatility weights from the given folds ONLY (early-choose): w proportional to
    1/sd of each book's daily realised P&L (pp, by close date, zero-filled over the folds' span),
    normalised to sum to 1. Correlation is ignored (pre-declared, 026 addendum 3)."""
    daily = {}
    for t, p in pooled.items():
        tr = pd.DataFrame([x for x in p["trades"] if x["fold_id"] in fold_ids])
        pp = tr["profit_abs"].fillna(0) * tr["scale_pp_per_abs"].fillna(0)
        day = pd.to_datetime(tr["close_date"], utc=True).dt.floor("D")
        daily[t] = pp.groupby(day).sum()
    lo = min(s.index.min() for s in daily.values())
    hi = max(s.index.max() for s in daily.values())
    span = pd.date_range(lo, hi, freq="D")
    sig = {t: float(s.reindex(span, fill_value=0.0).std()) for t, s in daily.items()}
    inv = {t: 1.0 / v for t, v in sig.items()}
    tot = sum(inv.values())
    return {t: {"weight": inv[t] / tot, "daily_sd_pp": sig[t]} for t in pooled}


def _stakes(trades: list[dict], stake: float) -> np.ndarray:
    """Capital each trade held: its resized stake, else filled stake x weight, else the config
    stake x weight (fixtures / old archives)."""
    return np.array([t["stake"] if t.get("stake") is not None
                     else (t.get("stake_amount") or stake) * t.get("weight", 1.0)
                     for t in trades], dtype=float)


def committed(trades: list[dict], stake: float) -> float:
    """Peak capital in open positions (per-trade stakes summed over open trades), per fold."""
    df = pd.DataFrame(trades)
    df["o"] = pd.to_datetime(df["open_date"], utc=True)
    df["c"] = pd.to_datetime(df["close_date"], utc=True)
    df["s"] = _stakes(trades, stake)
    peak = 0.0
    for _, g in df.groupby("fold_id"):
        ev = pd.concat([pd.Series(g["s"].to_numpy(), index=g["o"]),
                        pd.Series(-g["s"].to_numpy(), index=g["c"])]).sort_index(
                            kind="stable")
        # closes before opens at the same timestamp (a pair's slot frees before re-entry)
        ev = ev.groupby(level=0).apply(lambda x: x.sort_values()).reset_index(drop=True)
        peak = max(peak, float(ev.cumsum().max()))
    return peak


def mean_committed(trades: list[dict], stake: float, fold_ids: set[str] | None = None) -> float:
    """Time-averaged capital in open positions (per-trade stake x holding time / span). Max
    drawdown scales with it and Sharpe does not, so a drawdown verdict between arms of different
    mean exposure partly measures exposure (026 addendum 4)."""
    rows = [t for t in trades if fold_ids is None or t["fold_id"] in fold_ids]
    o = pd.to_datetime([t["open_date"] for t in rows], utc=True)
    c = pd.to_datetime([t["close_date"] for t in rows], utc=True)
    span = (c.max() - o.min()).total_seconds()
    return float((_stakes(rows, stake) * (c - o).total_seconds().to_numpy()).sum() / span)


def fit_weight(pooled: dict[str, dict], book: str, weights: dict[str, float], stake: float,
               capacity_: float, fold_ids: set[str], step: float = 0.05,
               top: float = 5.0) -> float:
    """Largest weight for `book` (grid `step`, other weights fixed) whose peak committed capital
    over `fold_ids` (early-choose) stays <= `capacity_`. Uncapped trades at w > 1 count at
    w x stake, the upper bound, so the fit is conservative."""
    best = 0.0
    for k in range(1, round(top / step) + 1):
        w = round(k * step, 10)
        c = combine(pooled, 1.0, {**weights, book: w}, stake)
        if committed([t for t in c["trades"] if t["fold_id"] in fold_ids], stake) > capacity_:
            break
        best = w
    return best


def shared_wallet(books: dict, capacity_: float, stake: float,
                  min_stake: float = MIN_STAKE) -> dict:
    """Replay a `combine` result through ONE wallet of `capacity_` (026 addendum 6).

    Per fold, in time order (closes before opens at one timestamp; same-time opens in book
    order, then pair), each entry fills min(its stake, capacity - committed), as freqtrade's
    `Wallets.validate_stake_amount` shrinks a stake to the available balance; below `min_stake`
    it is refused. Profit scales with the fill. APPROXIMATE: a refused or shrunk trade is
    otherwise replayed as recorded -- a refused entry would have left its pair free for a later
    signal that the archive does not hold."""
    order = {b: i for i, b in enumerate(dict.fromkeys(t["book"] for t in books["trades"]))}
    kept, delta = [], {}
    stats = {"refused": {}, "shrunk": {}, "refused_pp": 0.0, "shrunk_pp": 0.0}
    by_fold: dict = {}
    for t in books["trades"]:
        by_fold.setdefault(t["fold_id"], []).append(t)
    for fid, trades in by_fold.items():
        s_ = _stakes(trades, stake)
        ev = []
        for i, t in enumerate(trades):
            ev.append((pd.Timestamp(t["open_date"]), 1, order[t["book"]], t["pair"], i))
            ev.append((pd.Timestamp(t["close_date"]), 0, 0, "", i))
        held: dict[int, float] = {}
        for _, kind, _, _, i in sorted(ev):
            t = trades[i]
            if kind == 0:
                held.pop(i, None)
                continue
            room = capacity_ - sum(held.values())
            pp = (t["profit_abs"] or 0.0) * (t["scale_pp_per_abs"] or 0.0)
            if room < min_stake:
                stats["refused"][t["book"]] = stats["refused"].get(t["book"], 0) + 1
                stats["refused_pp"] += pp
                delta[fid] = delta.get(fid, 0.0) - pp
                continue
            fill = min(s_[i], room)
            held[i] = fill
            m = fill / s_[i]
            if m < 1:
                stats["shrunk"][t["book"]] = stats["shrunk"].get(t["book"], 0) + 1
                stats["shrunk_pp"] += pp * (1 - m)
                delta[fid] = delta.get(fid, 0.0) - pp * (1 - m)
            kept.append({**t, "profit_abs": t["profit_abs"] * m, "stake": fill})
    per_fold = [{**f, "return_pct": (None if f["return_pct"] is None
                                     else f["return_pct"] + delta.get(f["fold_id"], 0.0)),
                 "n_trades": sum(t["fold_id"] == f["fold_id"] for t in kept)}
                for f in books["per_fold"]]
    return {**books, "per_fold": per_fold, "trades": kept, "n_trades_pooled": len(kept),
            "shared_wallet": stats}


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
    ap.add_argument("--weights", default=None,
                    help="'risk' (inverse daily-P&L vol on f00-f07) or book=w,book=w")
    ap.add_argument("--fit", default=None,
                    help="book whose weight is fitted on f00-f07 to peak committed <= capacity")
    ap.add_argument("--shared-wallet", action="store_true",
                    help="replay the books through one wallet of stake x max_open_trades")
    ap.add_argument("--config", default=str(STRAT / "config" / "config_wf.json"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from evaluation import risk
    from evaluation.fingerprint import load_config
    from evaluation.halves import split

    tags = [t for t in a.books.split(",") if t]
    pooled = {t: _pooled(t) for t in tags}
    weights, chosen = None, None
    if a.weights == "risk":
        chosen = risk_weights(pooled, {f"f{i:02d}" for i in range(8)})
        weights = {t: v["weight"] for t, v in chosen.items()}
    elif a.weights:
        weights = {k: float(v) for k, v in (x.split("=") for x in a.weights.split(","))}
    cfg = load_config(a.config)
    stake = float(cfg["stake_amount"])
    early_ids = {f"f{i:02d}" for i in range(8)}
    if a.fit:
        weights = {**{t: 1.0 for t in tags}, **(weights or {})}
        weights[a.fit] = fit_weight(pooled, a.fit, weights, stake,
                                    stake * int(cfg["max_open_trades"]), early_ids)
    books = combine(pooled, a.stake_scale, weights, stake)
    if a.shared_wallet:
        books = shared_wallet(books, stake * int(cfg["max_open_trades"]), stake)
    bar = _pooled(a.bar)
    late = split(books, bar)["late"]
    whole = split(books, bar, split_after="f14")["early"]
    slots = round(int(cfg["max_open_trades"]) / a.stake_scale)
    cap = capacity(books["trades"], slots)
    late_ids = {f"f{i:02d}" for i in range(8, 15)}
    for name, ids in (("early", early_ids), ("late", late_ids)):
        cap[f"peak_committed_{name}"] = round(
            committed([t for t in books["trades"] if t["fold_id"] in ids], stake), 1)
    cap["bar_capacity"] = stake * int(cfg["max_open_trades"])
    cap["mean_committed_late"] = round(mean_committed(books["trades"], stake, late_ids), 1)
    cap["bar_mean_committed_late"] = round(mean_committed(bar["trades"], stake, late_ids), 1)
    cap["stake_upper_bound_trades"] = sum(bool(t.get("stake_upper_bound"))
                                          for t in books["trades"])
    if a.shared_wallet:
        sw = books["shared_wallet"]
        cap["shared_wallet"] = {**sw, "refused_pp": round(sw["refused_pp"], 2),
                                "shrunk_pp": round(sw["shrunk_pp"], 2)}
    h = _halves(books)
    bs, ws = late, whole
    lines = [
        f"## Books: {' + '.join(f'`{t}`' for t in tags)} (stake x{a.stake_scale:g}, {slots} slots"
        + (", ONE shared wallet)" if a.shared_wallet else ")")
        + (f", weights {json.dumps({t: round(v, 4) for t, v in weights.items()})}"
           if weights else ""),
        "",
        f"*{LABEL}*",
        "",
        *([f"risk weights chosen on f00-f07: {json.dumps(chosen, default=str)}", ""]
          if chosen else []),
        (f"total {h['total']:+.2f} pp, positive folds {h['positive']}/15, "
         f"early {h['early']:+.2f}, late {h['late']:+.2f}"),
        "",
        (f"late (f08-f14) vs `{a.bar}`: **{bs.get('verdict')}**, "
         f"{json.dumps({k: bs.get(k) for k in ('delta_pp', 'ci_pp_yr', 'p')}, default=str)}"),
        "",
        (f"whole span vs `{a.bar}`: **{ws.get('verdict')}**, "
         f"{json.dumps({k: ws.get(k) for k in ('delta_pp', 'ci_pp_yr', 'p')}, default=str)}"),
        "",
        f"late {risk.line(bs['risk'])}",
        "",
        f"whole span {risk.line(ws['risk'])}",
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
