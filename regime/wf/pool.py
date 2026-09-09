"""Pool per-fold trades into one trade set, retaining fold identity (FR-006, FR-012, FR-020).

Per-trade rows come from the backtest ARCHIVE each arm record names in `result_file` — freqtrade
silently ignores --export-filename, so user_data/backtest_results/*.zip is the only per-trade
source. The extraction is done by `regime.tail.compare._trades`, REUSED rather than reimplemented
(FR-013). That is a private symbol in a module this feature must not modify, so `smoke_test()`
below asserts it still works; if compare.py ever changes, this fails loudly instead of silently
pooling nothing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
STRAT_DIR = ROOT / "user_data/strategies"
sys.path.insert(0, str(STRAT_DIR))

from regime.tail.compare import _trades, _scale        # noqa: E402  (deliberate reuse, FR-013)

ARMS = STRAT_DIR / "regime/arms"


def smoke_test() -> bool:
    """T016a — prove the private coupling works before relying on it."""
    for p in sorted(ARMS.glob("*.json")):
        try:
            rec = json.loads(p.read_text())
        except Exception:
            continue
        if rec.get("returncode") == 0 and rec.get("n_trades") and rec.get("result_file"):
            rows = _trades(rec)
            if rows:
                print(f"  smoke test OK — compare._trades() returned {len(rows)} rows "
                      f"for {p.name}")
                return True
    print("  *** smoke test FAILED — compare._trades() returned nothing for any arm record")
    return False


def arm_record(tag: str) -> dict | None:
    for p in ARMS.glob(f"{tag}_*.json"):
        try:
            return json.loads(p.read_text())
        except Exception:
            return None
    return None


def pool(plan: dict) -> dict:
    pooled, per_fold, missing = [], [], []
    for f in plan["folds"]:
        rec = arm_record(f["arm_tag"])
        if not rec or rec.get("returncode") != 0:
            missing.append(f["fold_id"]); continue
        n = rec.get("n_trades") or 0
        if n == 0:
            # FR-012: zero-trade is a completed fold with no trades, NEVER a zero return.
            per_fold.append({"fold_id": f["fold_id"], "n_trades": 0, "return_pct": None,
                             "test_days": f["test_days"], "status": "zero_trade"})
            continue
        rows = _trades(rec)
        if rows is None:
            missing.append(f["fold_id"]); continue
        scale = _scale(rec, rows)
        for t in rows:
            pooled.append({
                "fold_id": f["fold_id"], "pair": t["pair"],
                "open_date": t["open_date"], "close_date": t.get("close_date"),
                "profit_abs": t["profit_abs"], "profit_ratio": t.get("profit_ratio"),
                "exit_reason": t.get("exit_reason"), "pair_group": "incumbent",
                "scale_pp_per_abs": scale,
            })
        per_fold.append({"fold_id": f["fold_id"], "n_trades": n,
                         "return_pct": rec.get("wallet_profit_pct"),
                         "test_days": f["test_days"], "status": "evaluated"})
    complete = len(per_fold)
    return {
        "run_id": plan["run_id"], "family": plan["family"],
        "params_sha": plan.get("params_sha"),
        "folds_planned": len(plan["folds"]), "folds_complete": complete,
        "folds_missing": missing,
        # FR-020: a partial set must never be presented as the full plan's result.
        "status": "complete" if not missing else "partial",
        "n_trades_pooled": len(pooled),
        "per_fold": per_fold, "trades": pooled,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    print("coupling check (T016a):")
    if not smoke_test():
        return 1
    plan = json.loads(Path(a.plan).read_text())
    out = pool(plan)
    Path(a.out).write_text(json.dumps(out, indent=2) + "\n")
    print(f"\npooled {out['n_trades_pooled']} trades from "
          f"{out['folds_complete']}/{out['folds_planned']} folds  status={out['status'].upper()}")
    if out["folds_missing"]:
        print(f"  MISSING folds (report is PARTIAL): {', '.join(out['folds_missing'])}")
    for pf in out["per_fold"]:
        r = "  (zero-trade)" if pf["status"] == "zero_trade" else f"{pf['return_pct']:+8.3f}%"
        print(f"  {pf['fold_id']}  {pf['n_trades']:>4}tr {r}  {pf['test_days']}d")
    print(f"written: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
