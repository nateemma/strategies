"""Arm-vs-control comparison for feature 003 (T008-T010).

Renders the five-part report from contracts/study-cli.md section 4 IN ORDER, and
encodes the verdict rule so the study cannot accidentally misread its own results:

  * W1 trades_hash identical to the control => `inconclusive_absorbed`, which
    OUTRANKS every other classification regardless of returns (FR-015). Six of six
    prior model-side changes in this codebase were absorbed by the entry gate, so
    this is the expected outcome, not an edge case.
  * A W1 delta below the power threshold renders as "indistinguishable", never as
    "improved" (FR-017; research.md R9 -- the control has 97 W1 trades).
  * Classifier metrics are NEVER an input to the verdict (FR-014). research.md R7:
    EV pointed the wrong way by 43pp, precision tripled while returns halved, and
    MCC fell while returns rose.
  * W2/W3 lie INSIDE the training window and are excluded from acceptance
    (FR-013); they are reported and labelled.
"""
from __future__ import annotations

import json
from pathlib import Path

ARMS = Path("user_data/strategies/regime/arms")
OOS_WINDOW = "W1"
IN_SAMPLE = ("W2", "W3")
POWER_THRESHOLD_PP = 1.0        # research.md R9

VERDICTS = ("inconclusive_absorbed", "improved", "worsened", "unchanged")


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
    if abs(delta) < POWER_THRESHOLD_PP:
        return ("unchanged",
                f"W1 delta {delta:+.2f}pp is below the {POWER_THRESHOLD_PP}pp power "
                f"threshold on {c['n_trades']} control trades -- indistinguishable")
    return ("improved" if delta > 0 else "worsened", f"W1 delta {delta:+.2f}pp")


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
