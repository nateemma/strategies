"""Generate and evaluate the repeat-signal position-adjustment arm, fold by fold.

WEIGHT REUSE IS THE HAZARD THIS FILE EXISTS TO CLOSE. `BaseNNStrategy.get_model_path` keys off
`self.__class__.__name__` (Framework/BaseNNStrategy.py:333-338), so a new class name means a new
`saved_data/` directory -- and `evaluation/run_folds.py` records the consequence in its own
docstring: a class with no model of its own TRAINS INSIDE ITS OWN BACKTEST, fitting the
evaluation window, silently, with returncode 0.

Position adjustment changes no model input, so nothing needs retraining. Each arm class
therefore overrides `get_model_path` to the UNGATED fold's directory, and `assert_reuses_weights`
refuses to run unless those weights exist and no new directory was created.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]      # repo root (contains user_data/)
STRAT = ROOT / "user_data/strategies"
SAVED = STRAT / "saved_data"
NNNC = STRAT / "NNNC"
OUT = Path(__file__).resolve().parent / "out"
SCRATCH = Path("/tmp/posadj_scratch")

TEMPLATE = '''# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""{cls} — repeat-signal position-adjustment arm for fold {fold_id}. GENERATED, do not edit.

Reuses {base}'s trained weights: position adjustment changes no model input, so retraining
would introduce a difference this comparison is not trying to measure.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from {base} import {base}
from _position_adjust import RepeatSignalPositionAdjustMixin


class {cls}(RepeatSignalPositionAdjustMixin, {base}):

    def get_model_path(self) -> str:
        # The UNGATED fold's directory, never this class's own.
        return self.get_storage_location() + "{base}/" + "{base}.keras"
'''


def assert_reuses_weights(cls: str, base: str) -> None:
    d = SAVED / base
    if not (d.is_dir() and any(d.glob("*.safetensors"))):
        raise SystemExit(f"no trained weights at {d} -- refusing to run {cls}")
    own = SAVED / cls
    if own.exists():
        raise SystemExit(f"{own} exists: {cls} has its own model dir and would train in-backtest")


def generate(fold_id: str, base: str) -> tuple[str, Path, Path]:
    cls = f"{base}_PA"
    py = NNNC / f"{cls}.py"
    py.write_text(TEMPLATE.format(cls=cls, base=base, fold_id=fold_id))
    src_json = NNNC / f"{base}.json"
    js = NNNC / f"{cls}.json"
    d = json.loads(src_json.read_text())
    d["strategy_name"] = cls           # MUST match the class or freqtrade aborts
    js.write_text(json.dumps(d, indent=4) + "\n")
    return cls, py, js


def scratch_for(cls: str) -> Path:
    dst = SCRATCH / cls
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for f in NNNC.glob("*.py"):
        (dst / f.name).symlink_to(f.resolve())
    for f in NNNC.glob("*.json"):
        shutil.copy(f, dst / f.name)
    return dst


def run_backtest(cls: str, timerange: str, config: Path, eps: bool = False) -> dict:
    scratch = scratch_for(cls)
    log = Path(f"/tmp/posadj_{cls}.log")
    env = os.environ.copy()
    env["PYTHONPATH"] = ":".join([".", str(STRAT), str(scratch), str(STRAT / "utils")])
    cmd = [str(ROOT / ".venv/bin/freqtrade"), "backtesting",
           "--strategy", cls, "--strategy-path", str(scratch),
           "-c", str(config), "--timerange", timerange,
           "--cache", "none", "--export", "trades"]
    if eps:
        cmd.append("--enable-position-stacking")
    t0 = time.time() - 1
    with log.open("w") as fh:
        rc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
    return {"returncode": rc, "log": str(log), **parse(log), **liveness(t0)}


NUM = r"[-+]?\d[\d,]*\.?\d*"
BASE_STAKE = 900.0


def liveness(after: float) -> dict:
    """Principle IV -- prove the adds actually fired, from the trade archive.

    NOT via `nr_of_successful_entries`: that key is NOT PRESENT in the exported trade dicts, so
    a `.get(key, 1)` reads 1 for every trade and reports zero adds on an arm that is working.
    Measured 2026-09-15: it reported 0 adds while the same trades carried stakes up to 2699.73,
    i.e. exactly the capped 2 adds on a 900 base. An inert-looking arm whose P&L had moved by
    6pp is the tell that the liveness probe, not the arm, was broken.

    `stake_amount` on a backtest trade is the TOTAL across fills, so the add count is the stake
    expressed in base-stake units, less one.
    """
    import glob
    import zipfile

    import numpy as np

    zs = [z for z in glob.glob(str(ROOT / "user_data/backtest_results/*.zip"))
          if os.path.getmtime(z) >= after]
    if not zs:
        return {"adds": None, "trades_with_adds": None}
    z = max(zs, key=os.path.getmtime)
    with zipfile.ZipFile(z) as zf:
        name = [n for n in zf.namelist() if n.endswith(".json") and "config" not in n][0]
        d = json.loads(zf.read(name))
    st = list(d["strategy"].keys())[0]
    tr = d["strategy"][st]["trades"]
    if not tr:
        return {"adds": 0, "trades_with_adds": 0, "archive": z}
    stakes = np.array([t["stake_amount"] for t in tr], dtype=float)
    adds = np.maximum(np.round(stakes / BASE_STAKE) - 1, 0).astype(int)
    return {"adds": int(adds.sum()), "trades_with_adds": int((adds > 0).sum()),
            "at_cap": int((adds >= 2).sum()), "archive": z}


def parse(log: Path) -> dict:
    t = log.read_text()
    def grab(label, cast=float):
        m = re.search(rf"│\s*{label}\s*│\s*({NUM})", t)
        return cast(m.group(1).replace(",", "")) if m else None
    m = re.search(r"│\s*Total/Daily Avg Trades\s*│\s*(\d+)\s*/", t)
    return {
        "n_trades": int(m.group(1)) if m else None,
        "profit_pct": grab("Total profit %"),
        "profit_abs": grab("Absolute profit"),
        "sharpe": grab(r"Sharpe \(closed trades\)"),
        "calmar": grab(r"Calmar \(closed trades\)"),
        "dd_account": grab("Max % of account underwater"),
        "dd_balance": grab(r"Max % of account underwater \(balance\)"),
        "rejected": grab("Rejected Entry signals", int),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", default=str(STRAT / "evaluation/folds/plan_nnnc.json"))
    ap.add_argument("--folds", help="comma-separated fold_ids; default all")
    ap.add_argument("--control", action="store_true", help="also re-run the ungated control")
    a = ap.parse_args()

    plan = json.loads(Path(a.plan).read_text())
    config = ROOT / plan["config"] if not Path(plan["config"]).is_absolute() else Path(plan["config"])
    folds = plan["folds"]
    if a.folds:
        keep = set(a.folds.split(","))
        folds = [f for f in folds if f["fold_id"] in keep]

    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for f in folds:
        base = f["strategy_class"]
        t0 = time.time()
        cls, _, _ = generate(f["fold_id"], base)
        assert_reuses_weights(cls, base)
        rec = {"fold_id": f["fold_id"], "base": base, "cls": cls,
               "timerange": f["test_timerange"]}
        rec["arm"] = run_backtest(cls, f["test_timerange"], config)
        if a.control:
            rec["control"] = run_backtest(base, f["test_timerange"], config)
        # The arm must not have created its own weights directory.
        if (SAVED / cls).exists():
            raise SystemExit(f"{cls} created {SAVED/cls} -- it trained inside its own backtest")
        results.append(rec)
        arm = rec["arm"]
        print(f"[{time.strftime('%H:%M:%S')}] {f['fold_id']}: "
              f"n={arm['n_trades']} adds={arm['adds']} profit={arm['profit_pct']}% "
              f"dd_bal={arm['dd_balance']} ({time.time()-t0:.0f}s)", flush=True)

    (OUT / "arms.json").write_text(json.dumps(results, indent=2))
    print(f"WROTE {OUT/'arms.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
