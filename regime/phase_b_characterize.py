"""Phase B - actionability characterisation (T014).

Contract: specs/001-market-phase-entry-gate/contracts/study-cli.md

Per instrument, per era: state distribution, median dwell time, and the state
transition matrix. Then eliminates instruments that cannot be acted on.

TWO KILL CRITERIA (FR-010, FR-011):

  DROP_SPAN      >= 25% of the host's REAL trades span a phase change between
                 entry and exit. This is the criterion's intent measured
                 DIRECTLY against the trade list, replacing the median-dwell
                 proxy used in the first draft.

                 The proxy was badly wrong. asset_trend has the LOWEST span rate
                 of any instrument (8.9%) yet median run length eliminated it,
                 while btc_exp4 -- where nearly half of trades span a regime
                 change (48.9%) -- passed. Median run length punishes
                 instruments that chop briefly around a boundary and rewards
                 ones with rare long runs, neither of which is the question.
                 Median dwell is still reported, as a diagnostic only.

  DROP_UNSTABLE  comparing W1 against W3 ONLY: max |change in state share| >
                 15pp, or the most-likely next state differs for >= half the
                 states. W1-vs-W2 CANNOT trigger this - those windows are three
                 months apart and share a regime (Constitution II).

--dwell-source has NO DEFAULT, deliberately. The threshold must come from the
host's own measured distribution; a figure borrowed from another strategy family
is exactly the error R6 records (the 1.4-31h range circulating in this project
belongs to the momentum basket, not to this host).

Run from the freqtrade root:

    PYTHONPATH=. .venv/bin/python user_data/strategies/regime/phase_b_characterize.py \\
        --dwell-source user_data/backtest_results/<baseline>.zip
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from regime import instruments as I
from regime.regimedata import UNKNOWN, TRADED_PAIRS, WINDOWS, window_slice

LABELS = Path(__file__).parent / "labels"
OUT_CSV = Path(__file__).parent / "actionability.csv"
CATALOGUE_CSV = Path(__file__).parent / "instrument_catalogue.csv"

# A state holding less than this share of a window's bars is reported with its
# sample count and excluded from verdict-driving comparisons (spec edge case).
MIN_STATE_SHARE = 0.05
# Fraction of real trades that may span a phase change before the instrument
# is unusable as an entry gate (spec FR-010 rationale).
MAX_SPAN_SHARE = 0.25
# Instability thresholds, W1 vs W3 only (FR-011).
MAX_SHARE_DRIFT = 0.15
BARS_PER_HOUR = 4


def span_rates(source: Path) -> tuple[dict, dict]:
    """Fraction of the host's real trades that span a phase change, per instrument.

    Direct measurement of what the dwell criterion was proxying: load the actual
    trades, look at each instrument's labels between entry and exit, and count
    how often the phase is not constant over the life of the trade.
    """
    from freqtrade.data.btanalysis import load_backtest_data
    trades = load_backtest_data(source)
    trades = trades.assign(_p=trades.pair.str.split("/").str[0])
    cache, out = {}, {}
    for inst in I.CATALOGUE:
        spans = tot = 0
        for _, t in trades.iterrows():
            fn = {"btc": "btc.feather", "market": "market.feather"}.get(
                inst.level, f"asset_{t._p}.feather")
            key = (fn, inst.name)
            if key not in cache:
                f = LABELS / fn
                cache[key] = (pd.read_feather(f).set_index("date")[inst.name].astype(str)
                              if f.exists() else None)
            s = cache[key]
            if s is None:
                continue
            seg = s.loc[(s.index >= t.open_date) & (s.index <= t.close_date)]
            seg = seg[seg != UNKNOWN]
            if len(seg) < 2:
                continue
            tot += 1
            spans += int(seg.nunique() > 1)
        out[inst.name] = (spans / tot) if tot else None
    return out, {"n_trades": int(len(trades))}


def host_dwell_threshold(source: Path) -> tuple[float, dict]:
    """25th percentile of the host's trade duration, in BARS."""
    from freqtrade.data.btanalysis import load_backtest_data
    trades = load_backtest_data(source)
    if trades is None or trades.empty:
        raise SystemExit(f"{source}: no trades - cannot derive a dwell threshold")
    h = (trades["close_date"] - trades["open_date"]).dt.total_seconds() / 3600.0
    p25 = float(h.quantile(0.25))
    return p25 * BARS_PER_HOUR, {
        "n_trades": int(len(trades)),
        "median_h": round(float(h.median()), 2),
        "p25_h": round(p25, 2),
        "p75_h": round(float(h.quantile(0.75)), 2),
    }


def dwell_runs(s: pd.Series) -> pd.Series:
    """Median consecutive-run length per state, in bars."""
    grp = (s != s.shift()).cumsum()
    runs = s.groupby(grp).agg(["first", "size"])
    return runs.groupby("first")["size"].median()


def transition_matrix(chunks: list) -> pd.DataFrame:
    """Row-stochastic next-state probabilities, aggregated across pairs.

    Counts are accumulated PER PAIR and only then normalised. Concatenating the
    pairs first and shifting would compute a bogus transition across every pair
    boundary (the last bar of one pair into the first bar of the next), and the
    pooled index carries duplicate timestamps besides.
    """
    total = None
    for c in chunks:
        nxt = c.shift(-1)
        ok = c.notna() & nxt.notna()
        if not ok.any():
            continue
        t = pd.crosstab(c[ok].reset_index(drop=True), nxt[ok].reset_index(drop=True))
        total = t if total is None else total.add(t, fill_value=0)
    if total is None:
        return pd.DataFrame()
    return total.div(total.sum(axis=1).replace(0, np.nan), axis=0)


def load_series(inst, pair: str | None) -> pd.Series | None:
    fname = {"btc": "btc.feather", "market": "market.feather"}.get(
        inst.level, f"asset_{pair}.feather")
    p = LABELS / fname
    if not p.exists():
        return None
    df = pd.read_feather(p).set_index("date")
    return df[inst.name].astype(str) if inst.name in df.columns else None


def characterise(inst, dwell_bars: float) -> tuple[list, dict]:
    """Rows for actionability.csv plus the per-era artefacts needed for the verdict."""
    rows, per_era = [], {}
    pairs = TRADED_PAIRS if inst.level == "asset" else [None]

    for era in WINDOWS:
        # Asset-level instruments are pooled across the traded pairs: the
        # instrument is one recipe, and eliminating it per-pair would be the
        # per-pair special-casing Constitution X forbids.
        chunks = []
        for p in pairs:
            s = load_series(inst, p)
            if s is None:
                continue
            w = window_slice(s, era)
            chunks.append(w[w != UNKNOWN])
        if not chunks:
            continue
        pooled = pd.concat(chunks)

        share = pooled.value_counts(normalize=True)
        dwell = pd.concat([dwell_runs(c) for c in chunks], axis=1).median(axis=1)
        trans = transition_matrix(chunks)
        per_era[era] = {"share": share, "dwell": dwell, "trans": trans,
                        "n_bars": int(len(pooled))}

        for st in share.index:
            rows.append({
                "instrument": inst.name, "level": inst.level, "era": era,
                "state": st,
                "share": round(float(share[st]), 4),
                "rare": bool(share[st] < MIN_STATE_SHARE),
                "median_dwell_bars": int(dwell.get(st, 0)),
                "median_dwell_hours": round(float(dwell.get(st, 0)) / BARS_PER_HOUR, 2),
                "n_bars": int(len(pooled)),
            })
    return rows, per_era


def verdict(inst, per_era: dict, span: float | None) -> tuple[str, str]:
    """SURVIVING / ELIMINATED with a measured, quoted reason."""
    if not per_era:
        return "ELIMINATED", "no labels found"

    # 1. Span - measured directly on the host's real trades.
    if span is not None and span >= MAX_SPAN_SHARE:
        return ("ELIMINATED",
                f"{span*100:.1f}% of real trades span a phase change (>= 25%)")

    # 2. Instability - W1 vs W3 ONLY (Constitution II).
    a, b = per_era.get("W1"), per_era.get("W3")
    if a is not None and b is not None:
        states = sorted(set(a["share"].index) | set(b["share"].index))
        drift = max(abs(float(a["share"].get(s, 0.0)) - float(b["share"].get(s, 0.0)))
                    for s in states)
        if drift > MAX_SHARE_DRIFT:
            return "ELIMINATED", f"state-share drift W1 vs W3 = {drift*100:.1f}pp > 15pp"

        flips, comparable = 0, 0
        for s in states:
            if s in a["trans"].index and s in b["trans"].index:
                comparable += 1
                if a["trans"].loc[s].idxmax() != b["trans"].loc[s].idxmax():
                    flips += 1
        if comparable and flips >= comparable / 2:
            return ("ELIMINATED",
                    f"most-likely next state flips for {flips}/{comparable} states, W1 vs W3")

    return "SURVIVING", ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dwell-source", required=True, type=Path,
                    help="host backtest archive; NO DEFAULT by design (R6)")
    ap.add_argument("--eras", default=",".join(WINDOWS))
    args = ap.parse_args()

    dwell_bars, host = host_dwell_threshold(args.dwell_source)
    print(f"host source : {args.dwell_source.name}")
    print(f"  {host['n_trades']} trades | median {host['median_h']}h | "
          f"p25 {host['p25_h']}h | p75 {host['p75_h']}h")
    I.validate_catalogue()
    spans, _ = span_rates(args.dwell_source)
    print(f"  => DROP_SPAN threshold = {MAX_SPAN_SHARE:.0%} of trades spanning "
          f"a phase change (measured directly)\n")
    all_rows, verdicts = [], []
    for inst in I.CATALOGUE:
        rows, per_era = characterise(inst, dwell_bars)
        all_rows += rows
        status, reason = verdict(inst, per_era, spans.get(inst.name))
        inst.status, inst.status_reason = status, reason
        w1 = per_era.get("W1")
        common = (w1["share"][w1["share"] >= MIN_STATE_SHARE].index if w1 is not None else [])
        verdicts.append({
            "instrument": inst.name, "level": inst.level,
            "states_used": len(common),
            "rare_states": int((w1["share"] < MIN_STATE_SHARE).sum()) if w1 is not None else 0,
            "span_pct": (round(spans[inst.name] * 100, 1)
                         if spans.get(inst.name) is not None else None),
            "median_dwell_h": (round(float(w1["dwell"].reindex(common).median()) / BARS_PER_HOUR, 2)
                               if w1 is not None and len(common) else None),
            "status": status, "reason": reason,
        })

    pd.DataFrame(all_rows).to_csv(OUT_CSV, index=False)
    I.to_frame().to_csv(CATALOGUE_CSV, index=False)

    v = pd.DataFrame(verdicts)
    print(v.to_string(index=False))
    print(f"\nwrote {OUT_CSV} and {CATALOGUE_CSV}")

    surv = v[v.status == "SURVIVING"]
    print(f"\nSURVIVING: {len(surv)} of {len(v)}")
    if surv.empty:
        print("\n*** ZERO instruments survive - the feature stops here and that is "
              "the finding (US1 checkpoint). ***")
        return 1
    print("  " + ", ".join(surv.instrument))
    return 0


if __name__ == "__main__":
    sys.exit(main())
