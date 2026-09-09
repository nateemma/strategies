"""Arm-vs-control comparison for feature 003 (T008-T010).

Renders the five-part report from contracts/study-cli.md section 4 IN ORDER, and
encodes the verdict rule so the study cannot accidentally misread its own results:

  * W1 trades_hash identical to the control => `inconclusive_absorbed`, which
    OUTRANKS every other classification regardless of returns (FR-015). Six of six
    prior model-side changes in this codebase were absorbed by the entry gate, so
    this is the expected outcome, not an edge case.
  * The verdict comes from a PAIRED BOOTSTRAP over the per-trade P&L, not a flat
    threshold. The old POWER_THRESHOLD_PP = 1.0 was a rule of thumb that was never
    derived: measured 2026-09-09, the NNNC W1 control's own bootstrap 95% CI is
    [2.383%, 17.649%] on 99 trades -- a +/-7.6pp half-width, so the 1.0pp rule was
    ~7.6x too lenient and passed noise as signal. A +1.86pp delta it called IMPROVED
    bootstraps to [-4.29, +7.93]pp, p=0.54. If the CI spans zero, the delta is NOT
    RESOLVED, whatever its sign or size.
  * Classifier metrics are NEVER an input to the verdict (FR-014). research.md R7:
    EV pointed the wrong way by 43pp, precision tripled while returns halved, and
    MCC fell while returns rose.
  * W2/W3 lie INSIDE the training window and are excluded from acceptance
    (FR-013); they are reported and labelled.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np

ARMS = Path("user_data/strategies/regime/arms")
BT_RESULTS = Path("user_data/backtest_results")
OOS_WINDOW = "W1"
IN_SAMPLE = ("W2", "W3")

MIN_DIFF_TRADES = 20            # below this the percentile bootstrap is degenerate:
                                # with 1 differing trade it resamples the same value
                                # every time, variance collapses to 0 and it reports
                                # p=0.000 on a +0.01pp delta. Too few items to
                                # estimate a sampling distribution is a POWER limit,
                                # not a null and certainly not a win.
N_RESAMPLES = 20000
BOOTSTRAP_SEED = 0              # fixed: the verdict must be reproducible
CI = (2.5, 97.5)

VERDICTS = ("inconclusive_absorbed", "improved", "worsened",
            "unchanged", "unresolvable", "inconclusive_underpowered")


def _trades(rec: dict) -> list | None:
    """Per-trade rows for an arm record, from the archive it names.

    run_arm.py stores `result_file`; --export-filename is deprecated and silently
    ignored by freqtrade, so the archive under user_data/backtest_results/ is the
    only per-trade source. Returns None if the archive has been pruned.
    """
    rf = rec.get("result_file")
    if not rf:
        return None
    z = BT_RESULTS / rf
    if not z.exists():
        return None
    try:
        with zipfile.ZipFile(z) as zf:
            names = [n for n in zf.namelist()
                     if n.endswith(".json") and "config" not in n]
            if not names:
                return None
            data = json.loads(zf.read(names[0]))
        return data["strategy"][rec["strategy"]]["trades"]
    except Exception:
        return None


def _scale(rec: dict, trades: list) -> float:
    """pp per unit of profit_abs, self-calibrated from the record itself.

    Avoids assuming a 10,000 starting balance. Verified exact on NNNC W1
    (983.1 abs -> 9.831%).
    """
    tot = sum(t["profit_abs"] for t in trades)
    if tot and rec.get("wallet_profit_pct"):
        return rec["wallet_profit_pct"] / tot
    return 100.0 / 10000.0


def bootstrap_delta(arm: dict, ctl: dict) -> dict | None:
    """Paired bootstrap of the arm-minus-control return difference.

    Pairs on (pair, open_date). Trades present and IDENTICAL in both contribute
    exactly zero and are excluded, which is what makes this tighter than comparing
    two independent totals -- for an exit-side change sharing every entry, only the
    handful of genuinely differing trades carry variance. Trades unique to one side
    enter with their own sign (the membership/symmetric-difference case, correct
    when the entry set itself moves).
    """
    at, ct = _trades(arm), _trades(ctl)
    if at is None or ct is None:
        return None
    scale = _scale(ctl, ct)
    key = lambda t: (t["pair"], t["open_date"])
    a = {key(t): t["profit_abs"] for t in at}
    c = {key(t): t["profit_abs"] for t in ct}
    shared = set(a) & set(c)
    identical = sum(1 for k in shared if abs(a[k] - c[k]) < 1e-9)
    diffs = [a[k] - c[k] for k in shared if abs(a[k] - c[k]) >= 1e-9]
    diffs += [a[k] for k in set(a) - set(c)]
    diffs += [-c[k] for k in set(c) - set(a)]
    if not diffs:
        return {"n_diff": 0, "identical": identical, "observed": 0.0,
                "lo": 0.0, "hi": 0.0, "p": 1.0, "resolved": False}
    d = np.asarray(diffs, dtype=float)
    n = len(d)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot = d[rng.integers(0, n, (N_RESAMPLES, n))].sum(axis=1) * scale
    lo, hi = np.percentile(boot, CI)
    p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
    return {"n_diff": n, "identical": identical, "observed": float(d.sum() * scale),
            "lo": float(lo), "hi": float(hi), "p": float(min(p, 1.0)),
            "resolved": bool(lo * hi > 0),
            "shared": len(shared), "arm_only": len(set(a) - set(c)),
            "ctl_only": len(set(c) - set(a))}


def load(tag: str, window: str) -> dict | None:
    p = ARMS / f"{tag}_{window}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return d if d.get("n_trades") else None


def verdict(arm_tag: str, ctl_tag: str) -> tuple[str, str]:
    """Return (verdict, reason). Absorption outranks everything (FR-015)."""
    a, c = load(arm_tag, OOS_WINDOW), load(ctl_tag, OOS_WINDOW)
    if a is None or c is None:
        return "unchanged", "missing arm or control record"
    if a["trades_hash"] == c["trades_hash"]:
        return ("inconclusive_absorbed",
                "W1 trades are byte-identical to the control -- the entry gate "
                "absorbed the change, so the returns are not evidence either way")
    delta = a["wallet_profit_pct"] - c["wallet_profit_pct"]
    bs = bootstrap_delta(a, c)
    if bs is None:
        return ("unresolvable",
                f"W1 delta {delta:+.2f}pp, but the per-trade archive for one side is "
                f"missing, so it CANNOT be tested. Re-run the arm. Do NOT read the "
                f"raw delta as a result -- the W1 noise floor is ~+/-7.6pp.")
    if bs["n_diff"] == 0:
        return ("inconclusive_absorbed",
                "every shared trade is identical and neither side has a unique "
                "trade -- nothing differs to test")
    if bs["n_diff"] < MIN_DIFF_TRADES:
        return ("inconclusive_underpowered",
                f"only {bs['n_diff']} trade(s) differ (delta {bs['observed']:+.2f}pp) "
                f"-- fewer than {MIN_DIFF_TRADES}, so the bootstrap cannot estimate a "
                f"sampling distribution. INCONCLUSIVE, not a result in either direction")
    if not bs["resolved"]:
        return ("unchanged",
                f"W1 delta {bs['observed']:+.2f}pp, 95% CI "
                f"[{bs['lo']:+.2f}, {bs['hi']:+.2f}]pp, p={bs['p']:.3f} over "
                f"{bs['n_diff']} differing trades -- CI spans zero, NOT RESOLVED")
    return ("improved" if bs["observed"] > 0 else "worsened",
            f"W1 delta {bs['observed']:+.2f}pp, 95% CI "
            f"[{bs['lo']:+.2f}, {bs['hi']:+.2f}]pp, p={bs['p']:.3f} over "
            f"{bs['n_diff']} differing trades -- resolved")


def report(arm_tag: str, ctl_tag: str, label: str = "", context: dict | None = None) -> str:
    out = [f"ARM {arm_tag}  vs  CONTROL {ctl_tag}" + (f"   [{label}]" if label else ""), ""]

    # 1. exit-config match -- a mismatch invalidates the comparison (FR-016)
    a1, c1 = load(arm_tag, OOS_WINDOW), load(ctl_tag, OOS_WINDOW)
    if a1 and c1:
        same = a1["exit_config_hash"] == c1["exit_config_hash"]
        out.append(f"1. exit_config_hash: arm {a1['exit_config_hash']} "
                   f"vs control {c1['exit_config_hash']} -> "
                   + ("MATCH" if same else "*** MISMATCH -- comparison INVALID "
                      "unless the hold window is deliberately under test ***"))
    out.append("")

    # 2. liveness per window (FR-015)
    out.append("2. trade-hash comparison (did the change reach the trades?)")
    for w in (OOS_WINDOW,) + IN_SAMPLE:
        a, c = load(arm_tag, w), load(ctl_tag, w)
        if not (a and c):
            out.append(f"     {w}: missing"); continue
        ident = a["trades_hash"] == c["trades_hash"]
        out.append(f"     {w}: arm {a['n_trades']:>4}tr  control {c['n_trades']:>4}tr  "
                   f"identical={ident}")
    out.append("")

    # 3. the verdict window
    a, c = load(arm_tag, OOS_WINDOW), load(ctl_tag, OOS_WINDOW)
    if a and c:
        d = a["wallet_profit_pct"] - c["wallet_profit_pct"]
        out.append(f"3. {OOS_WINDOW} — OUT-OF-SAMPLE (the verdict window)")
        out.append(f"     arm     {a['n_trades']:>4}tr  {a['wallet_profit_pct']:>7.2f}%  "
                   f"PF {a['profit_factor']:.2f}  win {a['win_rate']:.1f}%")
        out.append(f"     control {c['n_trades']:>4}tr  {c['wallet_profit_pct']:>7.2f}%  "
                   f"PF {c['profit_factor']:.2f}  win {c['win_rate']:.1f}%")
        out.append(f"     delta   {d:+.2f}pp")
        bs = bootstrap_delta(a, c)
        if bs is None:
            out.append("     paired bootstrap: UNAVAILABLE (per-trade archive pruned) "
                       "-- the delta above is untested")
        elif bs["n_diff"] and bs["n_diff"] < MIN_DIFF_TRADES:
            out.append(f"     paired bootstrap: only {bs['n_diff']} differing trade(s) "
                       f"-- under the {MIN_DIFF_TRADES} needed to estimate a "
                       f"distribution; INCONCLUSIVE")
        elif bs["n_diff"]:
            out.append(f"     paired bootstrap over {bs['n_diff']} differing trades "
                       f"({bs['shared']} shared, {bs['identical']} of them identical; "
                       f"{bs['arm_only']} arm-only, {bs['ctl_only']} control-only)")
            out.append(f"       delta {bs['observed']:+.2f}pp   95% CI "
                       f"[{bs['lo']:+.2f}, {bs['hi']:+.2f}]pp   p={bs['p']:.3f}   "
                       + ("RESOLVED" if bs["resolved"] else "NOT RESOLVED (CI spans 0)"))
    out.append("")

    # 4. in-sample windows, reported but excluded (FR-013)
    out.append("4. IN-SAMPLE (inside the training window — reported, NOT part of the verdict)")
    for w in IN_SAMPLE:
        a, c = load(arm_tag, w), load(ctl_tag, w)
        if not (a and c):
            out.append(f"     {w}: missing"); continue
        out.append(f"     {w}: arm {a['wallet_profit_pct']:>7.2f}%  "
                   f"control {c['wallet_profit_pct']:>7.2f}%  "
                   f"delta {a['wallet_profit_pct']-c['wallet_profit_pct']:+.2f}pp")
    out.append("")

    # 5. verdict
    v, why = verdict(arm_tag, ctl_tag)
    out.append(f"5. VERDICT: {v.upper()}")
    out.append(f"     {why}")
    if context:
        out.append("")
        out.append("   classifier metrics (CONTEXT ONLY — never an input to the verdict, FR-014):")
        out.append("     " + "  ".join(f"{k}={v2}" for k, v2 in context.items()))
    return "\n".join(out)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("usage: compare.py <arm_tag> <control_tag> [label]"); sys.exit(2)
    print(report(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else ""))
