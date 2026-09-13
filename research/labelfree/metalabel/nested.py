"""Nested walk-forward assignment: which trades may train the model for each fold (T022).

FR-007. The model evaluated on fold k trains only on trades from earlier folds.

KEYED ON close_date, NOT open_date (research R6). A trade that OPENED before fold k's
boundary but CLOSED after it carries an outcome that was not yet knowable at the boundary;
training on it leaks the future by exactly the width of its holding period. Keying on
close_date is strictly stronger and, with 90-day folds against a ~3h median hold, costs a
handful of rows.

Folds 0-4 are warm-up (91 trades) and are never evaluated. f05's model therefore trains on
those 91 and no more -- the design is at its thinnest exactly where it starts, which is why
n_train travels beside every per-fold figure (FR-014).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from research.labelfree import metalabel as M                 # noqa: E402
from research.labelfree.metalabel import archive as A         # noqa: E402


def build(rows: list[dict], plan: dict | None = None) -> list[dict]:
    """One assignment per evaluation fold, in fold order."""
    plan = plan or A.fold_ranges()
    out = []
    for fid in M.EVAL_FOLDS:
        boundary = pd.Timestamp(plan[fid]["test_start"], tz="UTC")
        train = [i for i, r in enumerate(rows)
                 if pd.Timestamp(r["close_date"]) < boundary]
        test = [i for i, r in enumerate(rows) if r["fold_id"] == fid]
        out.append({"eval_fold": fid, "test_start": plan[fid]["test_start"],
                    "train_idx": train, "test_idx": test,
                    "n_train": len(train), "n_test": len(test)})
    return out


def validate(rows: list[dict], assignment: list[dict]) -> list[str]:
    """Every violation, not the first -- a single leaked row and a systematic one look the
    same in a `raise`, and they are very different findings."""
    problems = []
    for a in assignment:
        boundary = pd.Timestamp(a["test_start"], tz="UTC")
        for i in a["train_idx"]:
            if pd.Timestamp(rows[i]["close_date"]) >= boundary:
                problems.append(
                    f"{a['eval_fold']}: training trade {rows[i]['pair']} closes "
                    f"{rows[i]['close_date']} at/after boundary {a['test_start']}")
        if a["eval_fold"] in M.WARMUP_FOLDS:
            problems.append(f"{a['eval_fold']} is a warm-up fold and must not be evaluated")
        if set(a["train_idx"]) & set(a["test_idx"]):
            problems.append(f"{a['eval_fold']}: train and test overlap")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", default="research/labelfree/metalabel/rows.jsonl")
    ap.add_argument("--out", default="research/labelfree/metalabel/assignment.json")
    a = ap.parse_args()

    rows = [json.loads(l) for l in Path(a.rows).read_text().splitlines() if l.strip()]
    asg = build(rows)
    problems = validate(rows, asg)

    print(f"NESTED ASSIGNMENT — {len(rows)} rows, {len(asg)} evaluation folds\n")
    print(f"  {'fold':<7}{'test_start':<13}{'n_train':>9}{'n_test':>8}")
    for x in asg:
        print(f"  {x['eval_fold']:<7}{x['test_start']:<13}{x['n_train']:>9}{x['n_test']:>8}")
    if problems:
        print(f"\n  *** {len(problems)} VIOLATIONS ***")
        for p in problems[:10]:
            print(f"    {p}")
        return 1
    print(f"\n  no training trade closes at or after its evaluation fold's start")
    Path(a.out).write_text(json.dumps(asg, indent=2))
    print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
