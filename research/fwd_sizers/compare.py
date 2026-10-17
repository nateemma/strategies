"""Forward confirmation read (preregistration.json): per arm, identity of trades with fwd_w_aro, arm
minus w_aro and minus cs_aro total pp, paired per-trade difference (USDT) with a bootstrap 90% interval,
agree share on the forward trades.

    PYTHONPATH=.:evaluation ../../.venv/bin/python -m research.fwd_sizers.compare
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ARMS = {"cs_cmp": ("fwd_cs_cmp_f15", "F15"), "cs_mix": ("fwd_cs_mix_f15", "F15")}
CMP = {"w_aro": ("fwd_w_aro_fwd1", "FWD1"), "cs_aro": ("fwd_cs_aro_fwd1", "FWD1")}


def _load(tag, fold):
    from evaluation.bootstrap import _trades
    from evaluation.pool import arm_record

    rec = arm_record(tag, fold)
    return rec, _trades(rec) or []


def _key(t):
    return (t["pair"], str(t["open_date"]), str(t["close_date"]))


def _total_pp(rec) -> float:
    for k in ("profit_total_pct", "total_profit_pct"):
        if k in rec:
            return float(rec[k])
    return float(rec["metrics"]["profit_total_pct"])


def main() -> int:
    rng = np.random.default_rng(0)
    ref = {k: _load(*v) for k, v in CMP.items()}
    out = {}
    for arm, (tag, fold) in ARMS.items():
        rec, tr = _load(tag, fold)
        r = {"n_trades": len(tr), "total_pp": _total_pp(rec)}
        for c, (crec, ctr) in ref.items():
            a, b = {_key(t): t for t in tr}, {_key(t): t for t in ctr}
            same = set(a) == set(b)
            diff = np.array([a[k]["profit_abs"] - b[k]["profit_abs"] for k in sorted(set(a) & set(b))])
            boots = [rng.choice(diff, len(diff)).sum() for _ in range(5000)] if len(diff) else [np.nan]
            r[f"vs_{c}"] = {"identical_trades": same, "only_arm": len(set(a) - set(b)), "only_cmp": len(set(b) - set(a)),
                            "delta_pp": r["total_pp"] - _total_pp(crec),
                            "paired_usdt": float(diff.sum()) if len(diff) else None,
                            "ci90_usdt": [float(np.percentile(boots, 5)), float(np.percentile(boots, 95))]}
        agree = [str(t.get("enter_tag", "")).endswith(":agree") for t in tr]
        r["agree_share"] = float(np.mean(agree)) if agree else None
        r["read"] = ("INCONCLUSIVE (trades differ from w_aro)" if not r["vs_w_aro"]["identical_trades"]
                     else "CONSISTENT" if r["vs_w_aro"]["delta_pp"] > 0 else "INCONSISTENT")
        out[arm] = r
        print(arm, json.dumps(r, default=float))
    (HERE / "compare.json").write_text(json.dumps(out, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
