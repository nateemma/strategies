"""Early/late halves split of a pooled walk-forward comparison (feature 023, research R13).

Feature 022 tested persistence by splitting its 15 folds into EARLY (f00-f07) and LATE
(f08-f14) by hand (`specs/022-perfold-guard-tuning/report.md`). No distant era exists for this
data (binanceus history is all selection or evaluation; `data/binance/` is empty), so this split
IS the persistence test the constitution's Principle II asks for -- and under quality gate 2 any
delta it supports is PROVISIONAL, not a finding. It is code now so every 023 report applies the
same cut the same way.

Each half is compared with `compare_pooled.compare_data`, the repository's one verdict
implementation, on that half's trades only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "user_data/strategies"))

from evaluation.compare_pooled import compare_data  # noqa: E402

PROVISIONAL = "PROVISIONAL -- not a finding under quality gate 2 (no distant era)"


def subset(pooled: dict, fold_ids: set[str]) -> dict:
    """A pooled set restricted to `fold_ids`, shaped like the original."""
    per_fold = [f for f in pooled["per_fold"] if f["fold_id"] in fold_ids]
    trades = [t for t in pooled["trades"] if t["fold_id"] in fold_ids]
    missing = [f for f in pooled.get("folds_missing", []) if f in fold_ids]
    return {**pooled, "per_fold": per_fold, "trades": trades,
            "n_trades_pooled": len(trades), "folds_complete": len(per_fold),
            "folds_planned": len(per_fold) + len(missing), "folds_missing": missing,
            "status": "complete" if not missing else "partial"}


def _side(pooled: dict) -> dict:
    rets = [f["return_pct"] for f in pooled["per_fold"] if f.get("return_pct") is not None]
    return {"folds": len(pooled["per_fold"]), "trades": pooled["n_trades_pooled"],
            "sum_pp": round(sum(rets), 3), "positive_folds": sum(1 for r in rets if r > 0)}


def split(arm: dict, ctl: dict, split_after: str = "f07") -> dict:
    """{early, late}: per-half per-fold sums, positive folds and a paired comparison."""
    folds = sorted({f["fold_id"] for f in ctl["per_fold"]} | {f["fold_id"] for f in arm["per_fold"]})
    early = {f for f in folds if f <= split_after}
    late = set(folds) - early
    out = {"split_after": split_after, "label": PROVISIONAL}
    for name, ids in (("early", early), ("late", late)):
        a, c = subset(arm, ids), subset(ctl, ids)
        cmp = compare_data(a, c) if c["trades"] else {"verdict": "UNRESOLVABLE"}
        out[name] = {"fold_ids": sorted(ids), "arm": _side(a), "ctl": _side(c),
                     "verdict": cmp["verdict"],
                     "delta_pp": cmp.get("delta_pp"), "ci_pp_yr": cmp.get("ci_pp_yr"),
                     "p": (cmp.get("bootstrap") or {}).get("p")}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", required=True, help="pooled JSON (evaluation/pool.py output)")
    ap.add_argument("--control", required=True, help="pooled JSON")
    ap.add_argument("--split-after", default="f07")
    a = ap.parse_args(argv)
    res = split(json.loads(Path(a.arm).read_text()), json.loads(Path(a.control).read_text()),
                a.split_after)
    print(res["label"])
    for half in ("early", "late"):
        h = res[half]
        print(f"  {half:<5} {h['fold_ids'][0]}-{h['fold_ids'][-1]}  "
              f"arm {h['arm']['sum_pp']:+8.3f}pp {h['arm']['positive_folds']}/{h['arm']['folds']}  "
              f"ctl {h['ctl']['sum_pp']:+8.3f}pp {h['ctl']['positive_folds']}/{h['ctl']['folds']}  "
              f"{h['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
