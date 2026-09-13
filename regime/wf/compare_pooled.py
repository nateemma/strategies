"""Arm-vs-control on POOLED walk-forward sets (FR-013).

Delegates to `regime.tail.compare.bootstrap_delta` — the SAME statistic used for single-window
arms, not a second implementation (research R8). Pooled sets pair on (fold_id, pair, open_date):
the same pair and timestamp can legitimately recur across folds, so the two-field key used for
single windows would collide.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "user_data/strategies"))

from regime.tail.compare import MIN_DIFF_TRADES, bootstrap_delta   # noqa: E402

KEY = ("fold_id", "pair", "open_date")


def compare(arm_path: str, ctl_path: str) -> str:
    a = json.loads(Path(arm_path).read_text())
    c = json.loads(Path(ctl_path).read_text())
    years = sum(f["test_days"] for f in c["per_fold"]) / 365.0
    # bootstrap_delta needs a record only for the pp scale; synthesise it from the pooled set.
    scale = c["trades"][0]["scale_pp_per_abs"]
    rec = {"wallet_profit_pct": sum(t["profit_abs"] for t in c["trades"]) * scale,
           "strategy": c["family"], "result_file": None}
    bs = bootstrap_delta(rec, rec, arm_trades=a["trades"], ctl_trades=c["trades"], key_fields=KEY)
    out = [f"POOLED ARM {Path(arm_path).name}  vs  CONTROL {Path(ctl_path).name}",
           f"  arm {a['n_trades_pooled']} trades / {a['folds_complete']} folds   "
           f"control {c['n_trades_pooled']} trades / {c['folds_complete']} folds",
           f"  pooled span {years:.2f} yr-equivalent"]
    if a["status"] != "complete" or c["status"] != "complete":
        out.append("  *** PARTIAL fold set — this is not the full plan's result (FR-020)")
    if bs is None:
        out.append("  UNRESOLVABLE — trades unavailable"); return "\n".join(out)
    if bs["n_diff"] == 0:
        out.append("  VERDICT: INCONCLUSIVE_ABSORBED — every trade is identical; nothing differs")
        return "\n".join(out)
    if bs["n_diff"] < MIN_DIFF_TRADES:
        out.append(f"  VERDICT: INCONCLUSIVE_UNDERPOWERED — only {bs['n_diff']} trade(s) differ")
        return "\n".join(out)
    ann = lambda v: v / years

    # ADMISSIBILITY (added 2026-09-13). The bootstrap above is untouched -- BOOTSTRAP_SEED
    # and the statistic are pinned and must stay comparable with every earlier run. This
    # only adds a refusal path, in the INCONCLUSIVE_* idiom this file already uses.
    #
    # THE TEST IS ABSOLUTE, NOT RELATIVE: if the ARM does not make money, a delta against
    # the control measures the CONTROL. Measured instances in the label-free study:
    #   brief 02  the P&L control lost 52% in a market that ROSE 72%, so every arm scored a
    #             large positive delta against it and 92% of the headline advantage turned
    #             out to be the gap to a broken comparator
    #   brief 04  the re-trained brief-1 comparator ran at exposure 0.0004 -- it abstained --
    #             and the raw delta read "WORSE" for an arm that actually traded
    # Neither is caught by a trade-count check, which is why this sits beside MIN_DIFF_TRADES
    # rather than inside it.
    arm_total = sum(t["profit_abs"] for t in a["trades"]) * scale
    ctl_total = sum(t["profit_abs"] for t in c["trades"]) * scale
    out.append(f"  arm total {arm_total:+.3f}pp   control total {ctl_total:+.3f}pp")
    if arm_total <= 0:
        out += [f"  VERDICT: INCONCLUSIVE_ARM_LOSES — the arm returns {arm_total:+.3f}pp "
                f"over the span, so it loses to holding cash.",
                f"  A delta against the control ({ann(bs['observed']):+.3f}pp/yr) measures "
                f"the control's losses, not the arm's value. Reported, not quoted."]
        return "\n".join(out)

    resolved = bs["resolved"]
    out += [f"  differing trades {bs['n_diff']}  ({bs['shared']} shared, {bs['identical']} identical;"
            f" {bs['arm_only']} arm-only, {bs['ctl_only']} control-only)",
            f"  delta {bs['observed']:+.3f}pp over the span  = {ann(bs['observed']):+.3f}pp/yr",
            f"  95% CI [{ann(bs['lo']):+.3f}, {ann(bs['hi']):+.3f}]pp/yr   p={bs['p']:.3f}",
            f"  VERDICT: {'IMPROVED' if resolved and bs['observed']>0 else 'WORSENED' if resolved else 'UNCHANGED'}"
            f"{'' if resolved else ' — CI spans zero, NOT RESOLVED'}"]
    return "\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--control", required=True)
    a = ap.parse_args()
    print(compare(a.arm, a.control))
