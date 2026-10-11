"""Feasibility gate for a reversion regime gate (feature 026 US1, contract C2, research R2/R3).

No backtests. For each pre-declared market-wide instrument (`utils.RegimeState`), on EXISTING
trades:

  1. choose a two-state rule on EARLY reversion trades only (f00-f07): threshold = the early-entry
     33rd/50th/67th percentile, "off" when value >= threshold or < threshold; options keeping < 50%
     of early trades "on" are excluded; pick the one minimising early "off" P&L (ties: higher
     on-share);
  2. score both halves; PASS iff
        off.early < 0  AND  off.late < 0  AND  on.early > 0  AND  on_share_early >= 0.5
        AND the late "off" is not one contiguous run over > 80% of late bars (calendar switch, M1).

PASS is decided on the reversion trades of the primary arm (`st_fam_combo`); the cross-check arm
(`st_gbb`) is scored with the SAME rule and reported, never a veto (M2). P&L unit = per-trade pp
contribution (profit_abs x scale_pp_per_abs), as in the 024/025 family columns. A trade's state is
the instrument's value at its signal bar (open_date - 15 min), as-of (causal).

A bucket split ignores substitution (023 Addendum 7): this only decides whether a gate is worth
testing as a walk-forward arm. PROVISIONAL under quality gate 2.

    python -m research.regime_gate.feasibility --primary st_fam_combo --cross st_gbb \\
        --out specs/026-reversion-regime-gate/feasibility.md

Exit: 0 at least one PASS, 3 none (STOP), 2 missing input.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


STRAT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(STRAT))

PERCENTILES = (33, 50, 67)
DIRECTIONS = ("ge", "lt")
MIN_ON_SHARE = 0.5
CALENDAR_SHARE = 0.8
EARLY = frozenset(f"f{i:02d}" for i in range(8))
LABEL = "PROVISIONAL -- not a finding under quality gate 2 (no distant era)"
PAIRS = ["ZEC", "XRP", "SOL", "LINK", "NEAR", "AAVE", "AVAX", "LTC", "BCH", "DOT"]


# --------------------------------------------------------------------------------- pure logic


def off_mask(values: pd.Series, rule: dict) -> pd.Series:
    v = pd.Series(values, dtype=float)
    off = v >= rule["threshold"] if rule["off_when"] == "ge" else v < rule["threshold"]
    return off & v.notna()  # NaN -> "on"


def choose_rule(trades: pd.DataFrame) -> dict | None:
    """Early-only rule choice (R2). Reads ONLY rows with half == 'early'."""
    early = trades[trades["half"] == "early"]
    vals = early["value"].dropna()
    if vals.empty:
        return None
    best = None
    for pct in PERCENTILES:
        thr = float(np.percentile(vals, pct))
        for d in DIRECTIONS:
            rule = {"percentile": pct, "threshold": thr, "off_when": d}
            off = off_mask(early["value"], rule)
            on_share = float((~off).mean())
            if on_share < MIN_ON_SHARE:
                continue
            key = (float(early.loc[off, "pnl"].sum()), -on_share)
            if best is None or key < best[0]:
                best = (key, {**rule, "on_share_early": round(on_share, 4)})
    return best[1] if best else None


def score(trades: pd.DataFrame, rule: dict) -> dict:
    off = off_mask(trades["value"], rule)
    out = {}
    for state, m in (("on", ~off), ("off", off)):
        for half in ("early", "late"):
            sel = m & (trades["half"] == half)
            out.setdefault("pnl", {}).setdefault(state, {})[half] = round(
                float(trades.loc[sel, "pnl"].sum()), 4
            )
            out.setdefault("trades", {}).setdefault(state, {})[half] = int(sel.sum())
    return out


def _passes(sc: dict, rule: dict, late_longest_run_share: float) -> tuple[bool, list[str]]:
    p = sc["pnl"]
    fails = []
    if not p["off"]["early"] < 0:
        fails.append("off.early >= 0")
    if not p["off"]["late"] < 0:
        fails.append("off.late >= 0")
    if not p["on"]["early"] > 0:
        fails.append("on.early <= 0")
    if not rule["on_share_early"] >= MIN_ON_SHARE:
        fails.append("on_share_early < 0.5")
    if late_longest_run_share > CALENDAR_SHARE:
        fails.append(f"calendar switch (late off run {late_longest_run_share:.0%} of late bars)")
    return not fails, fails


def evaluate(trades: pd.DataFrame, off_bar_stats: dict, rule: dict | None = None) -> dict:
    rule = rule if rule is not None else choose_rule(trades)
    if rule is None:
        return {"rule": None, "verdict": "STOP", "fails": ["no option keeps >= 50% on early"]}
    sc = score(trades, rule)
    ok, fails = _passes(sc, rule, off_bar_stats.get("late_longest_run_share", 0.0))
    return {"rule": rule, **sc, "verdict": "PASS" if ok else "STOP", "fails": fails}


def off_stats(series: pd.Series, rule: dict, start, end) -> dict:
    s = series[(series.index >= start) & (series.index < end)]
    if s.empty:
        return {"off_share": 0.0, "runs": 0, "longest_run_share": 0.0, "undefined": 0}
    off = off_mask(s, rule).to_numpy()
    runs, longest, cur = 0, 0, 0
    for o in off:
        if o:
            cur += 1
            if cur == 1:
                runs += 1
            longest = max(longest, cur)
        else:
            cur = 0
    n = len(off)
    return {
        "off_share": round(float(off.mean()), 4),
        "runs": runs,
        "longest_run_share": round(longest / n, 4),
        "undefined": int(s.isna().sum()),
    }


# ------------------------------------------------------------------------------------ loading


def _check_inputs(*tags):
    for t in tags:
        if not (STRAT / "evaluation" / "folds" / f"plan_{t}.json").is_file():
            raise FileNotFoundError(f"missing plan for {t}")


def _halves():
    plan = json.loads((STRAT / "evaluation/folds/plan_NNNC_A_Raw_Open.json").read_text())
    f = {x["fold_id"]: x for x in plan["folds"]}
    ts = lambda d: pd.Timestamp(d, tz="UTC")  # noqa: E731
    return {
        "early": (ts(f["f00"]["test_start"]), ts(f["f07"]["test_end"])),
        "late": (ts(f["f08"]["test_start"]), ts(f["f14"]["test_end"])),
    }


def _reversion_trades(tag: str, series: dict, family_filter: bool) -> dict[str, pd.DataFrame]:
    from evaluation.pool import pool
    from utils.FamilyExits import family_of

    pooled = pool(json.loads((STRAT / "evaluation/folds" / f"plan_{tag}.json").read_text()))
    tr = pd.DataFrame(pooled["trades"])
    if family_filter:
        tr = tr[tr["enter_tag"].map(family_of) == "reversion"]
    sig = pd.to_datetime(tr["open_date"], utc=True) - pd.Timedelta(minutes=15)
    base = pd.DataFrame(
        {
            "pnl": (tr["profit_abs"].fillna(0) * tr["scale_pp_per_abs"].fillna(0)).to_numpy(),
            "half": np.where(tr["fold_id"].isin(EARLY), "early", "late"),
        }
    )
    out = {}
    for inst, s in series.items():
        vals = s.sort_index().reindex(pd.DatetimeIndex(sig), method="ffill")
        out[inst] = base.assign(value=vals.to_numpy())
    return out


# ---------------------------------------------------------------------------------------- run


def run(primary: str, cross: str, out: Path) -> dict:
    from research.clfstudy.data import load_pair
    from utils import RegimeState as RS

    frames = {p: load_pair(f"{p}/USDT") for p in PAIRS}
    btc = load_pair("BTC/USDT")
    series = {inst: RS.compute(inst, frames, btc) for inst in RS.INSTRUMENTS}
    halves = _halves()
    prim = _reversion_trades(primary, series, family_filter=True)
    crs = _reversion_trades(cross, series, family_filter=False)
    res = {}
    for inst, s in series.items():
        rule = choose_rule(prim[inst])
        if rule is None:
            res[inst] = {"rule": None, "verdict": "STOP", "fails": ["no eligible option"]}
            continue
        bars = {h: off_stats(s, rule, *halves[h]) for h in halves}
        rec = evaluate(
            prim[inst], {"late_longest_run_share": bars["late"]["longest_run_share"]}, rule=rule
        )
        xc = evaluate(
            crs[inst], {"late_longest_run_share": bars["late"]["longest_run_share"]}, rule=rule
        )
        rec.update(
            {
                "bars": bars,
                "cross": {k: xc[k] for k in ("pnl", "trades", "verdict", "fails")},
                "cross_check_agrees": xc["verdict"] == rec["verdict"],
            }
        )
        res[inst] = rec
    _write(res, out, primary, cross)
    return res


def _write(res: dict, out: Path, primary: str, cross: str) -> None:
    out = Path(out)
    out.with_suffix(".json").write_text(json.dumps(res, indent=2, default=str) + "\n")
    L = [
        "# Feasibility: regime gate on the reversion family (026 US1)",
        "",
        f"*{LABEL}*",
        "",
        f"Primary (decides PASS): `{primary}` reversion trades. Cross-check: `{cross}`.",
        "",
        (
            "| instrument | rule (early-chosen) | on-share early | off pnl early / late "
            "| on pnl early / late | off trades e/l | off bars e/l | late longest off run "
            "| verdict | cross-check |"
        ),
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for inst, r in res.items():
        if r.get("rule") is None:
            L.append(f"| {inst} | none | — | — | — | — | — | — | STOP | — |")
            continue
        ru, p, t, b = r["rule"], r["pnl"], r["trades"], r["bars"]
        L.append(
            f"| {inst} | off if {ru['off_when']} p{ru['percentile']} ({ru['threshold']:.4g}) | "
            f"{ru['on_share_early']:.0%} | {p['off']['early']:+.2f} / {p['off']['late']:+.2f} | "
            f"{p['on']['early']:+.2f} / {p['on']['late']:+.2f} | {t['off']['early']} / "
            f"{t['off']['late']} | {b['early']['off_share']:.0%} / {b['late']['off_share']:.0%} | "
            f"{b['late']['longest_run_share']:.0%} | **{r['verdict']}** | {r['cross']['verdict']} |"
        )
    L += ["", "Failure reasons:", ""]
    for inst, r in res.items():
        L.append(f"- `{inst}`: {', '.join(r.get('fails') or []) or 'none (PASS)'}")
    n = sum(r["verdict"] == "PASS" for r in res.values())
    L += [
        "",
        f"**Overall: {'PROCEED to US2' if n else 'STOP'}** ({n} of {len(res)} instruments pass).",
        "",
        "Bucket splits ignore substitution (023 Addendum 7): a PASS only nominates an arm.",
    ]
    out.write_text("\n".join(L) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--primary", required=True)
    ap.add_argument("--cross", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    try:
        _check_inputs(a.primary, a.cross)
    except FileNotFoundError as exc:
        print(f"*** {exc}")
        return 2
    res = run(a.primary, a.cross, Path(a.out))
    return 0 if any(r.get("verdict") == "PASS" for r in res.values()) else 3


if __name__ == "__main__":
    sys.exit(main())
