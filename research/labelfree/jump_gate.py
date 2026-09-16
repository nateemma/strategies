"""Entry point for feature 016 / brief 08 -- the online sparse jump-model regime gate.

Named by the brief's starter prompt; the implementation lives in `research/labelfree/jump/`.

    python -m research.labelfree.jump_gate <command>

STATUS: the brief closed **KILL** at the T032 gate-activity checkpoint on 2026-09-15, before
any backtest. The gate blocks bars the incumbent's rvol >= 2.0 guard already rejects (14/15
folds) and which bars it blocks moves more with the random seed than the gate moves anything
(spread 0.115 vs block rate 0.104). The `leaktest`, `evaluate` and `drawdown` commands are
therefore NOT implemented: running them would produce numbers that cannot be interpreted, and
the kill checkpoint exists precisely to stop that. They refuse with the reason rather than
silently returning nothing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.labelfree.jump import OUT, SEEDS, assert_interpreter   # noqa: E402

KILLED_AT = "T032 gate-activity checkpoint, 2026-09-15"
BLOCKED_BY_KILL = {
    "leaktest": "US2 -- the smoothed-state leak measurement",
    "evaluate": "US4 -- the gated walk-forward arms",
    "drawdown": "US4 -- drawdown with bootstrap intervals",
    "check-weights": "US2 -- gated fold classes were never generated",
}


def _refuse(cmd: str) -> int:
    print(f"'{cmd}' ({BLOCKED_BY_KILL[cmd]}) is not available.\n"
          f"Brief 08 closed KILL at {KILLED_AT}.\n"
          f"A gate redundant with the incumbent rvol guard and unstable across seeds yields\n"
          f"uninterpretable performance numbers; see research/labelfree/jump/out/verdict.json\n"
          f"and research/label-free-study/briefs/results/08-2026-09-15.md", file=sys.stderr)
    return 2


def cmd_crosscheck(a) -> int:
    import numpy as np
    import pandas as pd

    from research.labelfree.jump import crosscheck as CC, features as FE, model as M

    audit = CC.audit_package_inference()
    print(json.dumps(audit, indent=2))
    if not audit["filtered"]:
        return 1
    f = FE.build(a.pair, pd.Timestamp(a.start, tz="UTC"), pd.Timestamp(a.end, tz="UTC"))
    x = f.to_numpy(float)
    fi = M.fit(x, 2, a.lam, seed=0, n_init=1)
    xs = (x - fi.scaler_mean) / fi.scaler_scale
    res = CC.matched_parameter_agreement(xs, 2, a.lam, seed=0)
    print(f"\nFR-016a matched     {res['matched']:.4f} -> {CC.verdict(res['matched'])}")
    print(f"FR-016b independent {res['independent']:.4f}  ({CC.describe_divergence(res)})")
    return 0 if CC.verdict(res["matched"]) == "agree" else 1


def cmd_seeds(a) -> int:
    import pandas as pd

    from research.labelfree.jump import features as FE, model as M

    f = FE.build(a.pair, pd.Timestamp(a.start, tz="UTC"), pd.Timestamp(a.end, tz="UTC"))
    x = f.to_numpy(float)
    fits = M.fit_ensemble(x, 2, a.lam, seeds=tuple(int(s) for s in a.seeds.split(",")))
    print(f"seed agreement {M.seed_agreement(fits, x):.4f}  (FR-023 floor 0.95)")
    return 0


def cmd_activity(a) -> int:
    from research.labelfree.jump import drive
    return drive.main()


def cmd_report(a) -> int:
    p = OUT / "activity.json"
    if not p.exists():
        print("no activity.json -- run `activity` first", file=sys.stderr)
        return 1
    d = json.loads(p.read_text())
    print(f"folds {d['n_folds']} | active {d['n_active']} | "
          f"redundant_with_rvol {d['n_redundant_with_rvol']} | "
          f"mean seed agreement {d['mean_seed_agreement']:.3f} | "
          f"mean seed spread {d['mean_across_seed_spread']:.3f}")
    v = OUT / "verdict.json"
    if v.exists():
        print(f"\nVERDICT: {json.loads(v.read_text())['verdict']} at {KILLED_AT}")
    return 0


def main(argv=None) -> int:
    assert_interpreter()
    ap = argparse.ArgumentParser(prog="jump_gate", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--pair", default="SOL/USDT")
        p.add_argument("--start", default="2023-01-01")
        p.add_argument("--end", default="2024-06-30")
        p.add_argument("--lam", type=float, default=3511.19)
        return p

    common(sub.add_parser("crosscheck", help="package audit + FR-016a/FR-016b")).set_defaults(fn=cmd_crosscheck)
    s = common(sub.add_parser("seeds", help="cross-seed state agreement (FR-022/023)"))
    s.add_argument("--seeds", default=",".join(str(x) for x in SEEDS))
    s.set_defaults(fn=cmd_seeds)
    sub.add_parser("activity", help="T031/T031a -- 15-fold gate activity + rvol redundancy").set_defaults(fn=cmd_activity)
    sub.add_parser("redundancy", help="alias of activity; redundancy is reported per fold").set_defaults(fn=cmd_activity)
    sub.add_parser("report", help="summarise out/activity.json and the verdict").set_defaults(fn=cmd_report)
    for c in BLOCKED_BY_KILL:
        sub.add_parser(c, help=f"UNAVAILABLE -- brief closed KILL at {KILLED_AT}").set_defaults(
            fn=lambda a, _c=c: _refuse(_c))

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
