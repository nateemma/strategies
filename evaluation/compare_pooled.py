"""Arm-vs-control on POOLED walk-forward sets (FR-013).

Delegates to `evaluation.bootstrap.bootstrap_delta` — the SAME statistic used for single-window
arms, not a second implementation (research R8). Pooled sets pair on (fold_id, pair, open_date):
the same pair and timestamp can legitimately recur across folds, so the two-field key used for
single windows would collide.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# parents: [0]=evaluation [1]=strategies [2]=user_data [3]=repo root (contains user_data/).
# Was parents[4] when this lived at regime/wf/; the 2026-09-14 promotion to evaluation/
# (a3c75bc) moved it one level shallower and the index was not decremented, so every
# path resolved one directory ABOVE the repo. Fixed 2026-10-07.
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "user_data/strategies"))

from evaluation.bootstrap import MIN_DIFF_TRADES, bootstrap_delta   # noqa: E402

KEY = ("fold_id", "pair", "open_date")


def compare_data(a: dict, c: dict) -> dict:
    """The comparison as data. `compare()` formats this; feature 023's twin and attribution
    tools consume it directly, so there is ONE verdict implementation, not two.

    `verdict` is one of: UNRESOLVABLE, INCONCLUSIVE_ABSORBED, INCONCLUSIVE_UNDERPOWERED,
    INCONCLUSIVE_ARM_LOSES, IMPROVED, WORSENED, UNCHANGED.
    """
    years = sum(f["test_days"] for f in c["per_fold"]) / 365.0
    out = {"arm_trades": a["n_trades_pooled"], "arm_folds": a["folds_complete"],
           "ctl_trades": c["n_trades_pooled"], "ctl_folds": c["folds_complete"],
           "years": years,
           "partial": a["status"] != "complete" or c["status"] != "complete"}
    # bootstrap_delta needs a record only for the pp scale; synthesise it from the pooled set.
    scale = c["trades"][0]["scale_pp_per_abs"]
    rec = {"wallet_profit_pct": sum(t["profit_abs"] for t in c["trades"]) * scale,
           "strategy": c["family"], "result_file": None}
    bs = bootstrap_delta(rec, rec, arm_trades=a["trades"], ctl_trades=c["trades"], key_fields=KEY)
    out["bootstrap"] = bs
    if bs is None:
        out["verdict"] = "UNRESOLVABLE"; return out
    if bs["n_diff"] == 0:
        out["verdict"] = "INCONCLUSIVE_ABSORBED"; return out
    if bs["n_diff"] < MIN_DIFF_TRADES:
        out["verdict"] = "INCONCLUSIVE_UNDERPOWERED"; return out

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
    out["arm_total_pp"] = sum(t["profit_abs"] for t in a["trades"]) * scale
    out["ctl_total_pp"] = sum(t["profit_abs"] for t in c["trades"]) * scale
    out["delta_pp"] = bs["observed"]
    out["delta_pp_yr"] = bs["observed"] / years
    out["ci_pp_yr"] = [bs["lo"] / years, bs["hi"] / years]
    if out["arm_total_pp"] <= 0:
        out["verdict"] = "INCONCLUSIVE_ARM_LOSES"; return out
    resolved = bs["resolved"]
    out["verdict"] = ("IMPROVED" if resolved and bs["observed"] > 0
                      else "WORSENED" if resolved else "UNCHANGED")
    return out


def compare(arm_path: str, ctl_path: str) -> str:
    a = json.loads(Path(arm_path).read_text())
    c = json.loads(Path(ctl_path).read_text())
    d = compare_data(a, c)
    years, bs = d["years"], d["bootstrap"]
    out = [f"POOLED ARM {Path(arm_path).name}  vs  CONTROL {Path(ctl_path).name}",
           f"  arm {a['n_trades_pooled']} trades / {a['folds_complete']} folds   "
           f"control {c['n_trades_pooled']} trades / {c['folds_complete']} folds",
           f"  pooled span {years:.2f} yr-equivalent"]
    if d["partial"]:
        out.append("  *** PARTIAL fold set — this is not the full plan's result (FR-020)")
    v = d["verdict"]
    if v == "UNRESOLVABLE":
        out.append("  UNRESOLVABLE — trades unavailable"); return "\n".join(out)
    if v == "INCONCLUSIVE_ABSORBED":
        out.append("  VERDICT: INCONCLUSIVE_ABSORBED — every trade is identical; nothing differs")
        return "\n".join(out)
    if v == "INCONCLUSIVE_UNDERPOWERED":
        out.append(f"  VERDICT: INCONCLUSIVE_UNDERPOWERED — only {bs['n_diff']} trade(s) differ")
        return "\n".join(out)
    ann = lambda v: v / years
    arm_total, ctl_total = d["arm_total_pp"], d["ctl_total_pp"]
    out.append(f"  arm total {arm_total:+.3f}pp   control total {ctl_total:+.3f}pp")
    if v == "INCONCLUSIVE_ARM_LOSES":
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
