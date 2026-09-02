"""Phase C - THE STOP GATE (T022).

Contract: specs/001-market-phase-entry-gate/contracts/study-cli.md
Design:   research R12 (per-trade, revised at T013), FR-018a/b/c/d

THE QUESTION. The conservative configuration's trades are a strict SUBSET of the
aggressive one's: it takes none of its own, it only declines some. Those declined
trades are the MARGINAL SET. Does the phase state in force at a marginal trade's
ENTRY predict whether that trade was worth taking?

THE COMPARATORS. For a marginal set M with realized per-trade returns:

    always-aggressive   takes all of M  ->  R_agg = sum(profit_ratio over M)
    always-conservative takes none      ->  R_con = 0
    phase rule          takes S c= M    ->  R_phase = sum over S

The bar is the BETTER FIXED CHOICE, max(R_agg, R_con) - never the tuned
status quo, which is neither available option (FR-018a).

FIT AND EVALUATION ARE DIFFERENT WINDOWS. The rule is: take the states whose
mean marginal return was positive in the FIT window; apply that fixed mapping to
the EVAL window. Fitting and scoring on one window is in-sample and proves
nothing - it is the shape of the day-of-week error. A PASS needs a temporally
distant fit/eval pair (Constitution II).

POWER. W3's marginal set holds 26 trades, which supports detecting a LARGE
effect only. The verdict therefore distinguishes "no effect detected" from "no
effect exists" (FR-018d).

Run from the freqtrade root:

    PYTHONPATH=. .venv/bin/python user_data/strategies/regime/phase_c_trackability.py
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from regime import instruments as I
from regime.regimedata import UNKNOWN, WINDOWS

HERE = Path(__file__).parent
LABELS = HERE / "labels"
ARMS = HERE / "arms"
OUT_CSV = HERE / "trackability.csv"
VERDICT_MD = HERE / "VERDICT.md"

# Arms produced at the study operating point (rvol = 2.0), aggressive/conservative.
ARM_TAGS = {("W1", "agg"): "g_rvol20", ("W1", "con"): "p_rvol20_t080",
            ("W2", "agg"): "pc069", ("W2", "con"): "pc080",
            ("W3", "agg"): "pc069", ("W3", "con"): "pc080"}
# Temporally distant pairs. W1<->W2 are three months apart and cannot establish
# persistence, so they are NOT admissible as a PASS pair (Constitution II).
DISTANT_PAIRS = [("W1", "W3"), ("W3", "W1"), ("W2", "W3"), ("W3", "W2")]
MIN_MARGINAL = 20


def load_arm_trades(tag: str, window: str) -> pd.DataFrame:
    from freqtrade.data.btanalysis import load_backtest_data
    rec = json.loads((ARMS / f"{tag}_{window}.json").read_text())
    return load_backtest_data(Path("user_data/backtest_results") / rec["result_file"])


def marginal_set(window: str) -> tuple[pd.DataFrame, float, dict]:
    """The trades the aggressive config takes and the conservative one declines.

    NOT a strict subset at the study operating point. Measured: the conservative
    arm takes 4/56, 14/136, 2/25 trades (7-10%) the aggressive one does not.
    The cause is NOT the position cap - peak concurrency is 5 against a limit of
    10, never reached - but freqtrade's one-open-trade-per-pair rule: while the
    aggressive config holds a position it cannot re-enter that pair, whereas the
    conservative config, having declined the earlier entry, is free to take the
    later one.

    So the comparison is between two realized policies rather than a set and its
    subset:
        M      = trades ONLY the aggressive config takes - what the phase rule
                 chooses among
        R_con  = the return of the trades ONLY the conservative config takes,
                 which is what declining all of M actually earns instead

    CAVEAT: a phase rule that declines part of M could unblock further trades of
    its own, which cannot be known without re-running the backtest per rule. The
    R_con offset is therefore an approximation of the conservative alternative,
    and it is small relative to M.
    """
    a = load_arm_trades(ARM_TAGS[(window, "agg")], window)
    c = load_arm_trades(ARM_TAGS[(window, "con")], window)
    key = ["pair", "open_date"]
    ka, kc = a[key].apply(tuple, axis=1), c[key].apply(tuple, axis=1)
    sa, sc = set(ka), set(kc)
    m = a[~ka.isin(sc)].copy()
    con_only = c[~kc.isin(sa)]
    m["p"] = m.pair.str.split("/").str[0]
    info = {"n_agg": len(a), "n_con": len(c), "n_marginal": len(m),
            "n_con_only": len(con_only),
            "violation_pct": round(len(con_only) / max(len(c), 1) * 100, 1)}
    return m, float(con_only.profit_ratio.sum()), info


def phase_at_entry(m: pd.DataFrame, inst) -> pd.Series:
    """The instrument's state on each marginal trade's entry bar."""
    out, cache = [], {}
    for _, t in m.iterrows():
        fname = {"btc": "btc.feather", "market": "market.feather"}.get(
            inst.level, f"asset_{t.p}.feather")
        if fname not in cache:
            f = LABELS / fname
            cache[fname] = (pd.read_feather(f).set_index("date") if f.exists() else None)
        df = cache[fname]
        if df is None or inst.name not in df.columns:
            out.append(UNKNOWN); continue
        s = df[inst.name].astype(str)
        idx = s.index[s.index <= t.open_date]
        out.append(s.loc[idx[-1]] if len(idx) else UNKNOWN)
    return pd.Series(out, index=m.index, name="state")


def fit_rule(m: pd.DataFrame, states: pd.Series) -> set:
    """States whose mean marginal return was positive in the fit window."""
    g = m.profit_ratio.groupby(states).mean()
    return set(g[g > 0].index) - {UNKNOWN}


def evaluate(m: pd.DataFrame, states: pd.Series, take: set, r_con: float) -> dict:
    r_agg = float(m.profit_ratio.sum())
    sel = states.isin(take)
    r_phase = float(m.profit_ratio[sel].sum())
    bar = max(r_agg, r_con)
    return {"n_marginal": int(len(m)), "n_taken": int(sel.sum()),
            "R_agg_pct": round(r_agg * 100, 3), "R_con_pct": round(r_con * 100, 3),
            "R_phase_pct": round(r_phase * 100, 3),
            "bar_pct": round(bar * 100, 3),
            "edge_pct": round((r_phase - bar) * 100, 3),
            "beats_bar": bool(r_phase > bar)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-marginal", type=int, default=MIN_MARGINAL)
    args = ap.parse_args()

    I.validate_catalogue()
    cat = pd.read_csv(HERE / "instrument_catalogue.csv")
    surviving = set(cat[cat.status == "SURVIVING"].name)
    insts = [i for i in I.CATALOGUE if i.name in surviving]
    print(f"surviving instruments: {len(insts)}  ({', '.join(sorted(surviving))})\n")

    margins, r_cons, phases = {}, {}, {}
    for w in WINDOWS:
        margins[w], r_cons[w], info = marginal_set(w)
        print(f"  {w}: agg={info['n_agg']} con={info['n_con']} -> marginal={info['n_marginal']}"
              f" (summed {margins[w].profit_ratio.sum()*100:+.2f}%);"
              f" con-only={info['n_con_only']} ({info['violation_pct']}%,"
              f" {r_cons[w]*100:+.2f}%)")
    print()

    rows = []
    for inst in insts:
        for w in WINDOWS:
            phases[(inst.name, w)] = phase_at_entry(margins[w], inst)
        for fit_w, eval_w in DISTANT_PAIRS:
            if len(margins[eval_w]) < args.min_marginal:
                continue
            take = fit_rule(margins[fit_w], phases[(inst.name, fit_w)])
            if not take:
                rows.append({"instrument": inst.name, "level": inst.level,
                             "fit": fit_w, "eval": eval_w, "rule": "(empty)",
                             "n_marginal": len(margins[eval_w]), "n_taken": 0,
                             "R_agg_pct": round(margins[eval_w].profit_ratio.sum()*100, 3),
                             "R_con_pct": round(r_cons[eval_w]*100, 3), "R_phase_pct": 0.0,
                             "bar_pct": round(max(margins[eval_w].profit_ratio.sum(), r_cons[eval_w])*100, 3),
                             "edge_pct": None, "beats_bar": False})
                continue
            res = evaluate(margins[eval_w], phases[(inst.name, eval_w)], take, r_cons[eval_w])
            rows.append({"instrument": inst.name, "level": inst.level,
                         "fit": fit_w, "eval": eval_w,
                         "rule": "|".join(sorted(take)), **res})

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(df[["instrument", "fit", "eval", "n_marginal", "n_taken",
              "R_agg_pct", "R_phase_pct", "bar_pct", "edge_pct",
              "beats_bar"]].to_string(index=False))

    # A PASS needs the SAME instrument to beat the bar on at least two distant
    # fit/eval pairs.
    wins = df[df.beats_bar].groupby("instrument").size()
    passers = wins[wins >= 2]

    lines = ["# Phase C verdict", ""]
    if len(passers):
        best = passers.idxmax()
        verdict = f"PASS {best}"
        lines += [f"**VERDICT: PASS — `{best}`**", "",
                  f"Beat the better fixed choice on {int(passers[best])} temporally "
                  "distant fit/evaluate pairs."]
    else:
        verdict = "FAIL"
        lines += ["**VERDICT: FAIL**", "",
                  "No surviving instrument beat the better fixed choice on two or more",
                  "temporally distant fit/evaluate pairs.", "",
                  "**This is 'no effect detected', not 'no effect exists'** (FR-018d).",
                  f"W3's marginal set holds {len(margins['W3'])} trades, which supports",
                  "detecting only a large effect; and outcomes are truncated near +3% by",
                  "the exit policy, compressing any magnitude signal."]
    lines += ["", "## Evidence", "", df.to_markdown(index=False), "",
              "## Marginal sets", ""]
    for w in WINDOWS:
        lines.append(f"- **{w}**: {len(margins[w])} trades, "
                     f"summed {margins[w].profit_ratio.sum()*100:+.2f}%")
    VERDICT_MD.write_text("\n".join(lines) + "\n")

    print(f"\nwrote {OUT_CSV} and {VERDICT_MD}")
    print(f"\nVERDICT: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
