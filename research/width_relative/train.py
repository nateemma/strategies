"""Train the width-relative folds, one at a time (033 T016).

The 032 driver (research/candidate_training/train.py) pointed at the `_sz` plans: per fold, write
the fold class from the params snapshot (no threshold), train if the fold's model key has no
weights, verify. Resumable; refuses to start while another chain runs. Both arm variants share the
model key, so training the `_sz` plan trains the `_szx` arm's weights too. After each trained fold,
prints its label audit (width_relative.json) next to row_filter.json.

    PYTHONPATH=. ../../.venv/bin/python -m research.width_relative.train --model ARO [--folds f00]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.candidate_training import train as T


PLANS = {"ARO": "wr_aro_sz", "GBR": "wr_gbr_sz"}


def summary(rep: dict | None) -> dict | None:
    if rep is None:
        return None
    shares = {s: [b["pos_share"] for b in rep["buckets"] if b["side"] == s] for s in ("buy", "sell")}
    return {"kept": rep["kept"], "invalid": rep["invalid"],
            "spearman_buy": rep.get("spearman_buy"), "spearman_sell": rep.get("spearman_sell"),
            "share_range_buy": [min(shares["buy"]), max(shares["buy"])] if shares["buy"] else None,
            "share_range_sell": [min(shares["sell"]), max(shares["sell"])] if shares["sell"] else None}


def run(model: str, folds: set[str] | None, seed: int = 1) -> list[dict]:
    T.PLANS.update(PLANS)  # the 032 driver, pointed at this feature's plans
    T.BUSY = tuple(T.BUSY) + ("research.width_relative.train ",)
    out = T.run(model, folds, seed)
    plan = json.loads((T.FOLDS / f"plan_{PLANS[model]}.json").read_text())
    for r in out:
        f = next(x for x in plan["folds"] if x["fold_id"] == r["fold"])
        p = Path(T._location(f, plan["base_strategy"], plan["config"], seed).path).parent
        wr = p / "width_relative.json"
        r["width_relative"] = summary(json.loads(wr.read_text()) if wr.is_file() else None)
        print(f"{r['fold']}: width_relative {json.dumps(r['width_relative'])}", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, choices=sorted(PLANS))
    ap.add_argument("--folds", help="comma-separated fold ids; default all")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    run(a.model, set(a.folds.split(",")) if a.folds else None, a.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
