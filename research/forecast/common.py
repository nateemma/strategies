"""038 shared constants, fold windows, costs and the pre-registration guard (research R2, R8, R9).

    PYTHONPATH=.:evaluation ../../.venv/bin/python -m research.forecast.<module>
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import pandas as pd

STRAT = Path(__file__).resolve().parents[2]
ROOT = STRAT.parents[1]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(STRAT), str(STRAT / "evaluation"), str(STRAT / "utils")]

OUT = ROOT / "user_data" / "data" / "forecast_038"
FRAMES = OUT / "frames"
PRED = OUT / "pred"
DATADIR = ROOT / "user_data" / "data" / "binanceus"
PREREG = HERE / "preregistration.json"
AMEND_HARMONIC = HERE / "preregistration_harmonic.json"  # post-hoc harmonic methods (038 follow-up)
HARMONIC = ("harm", "harm_s", "harm_d", "harm_sd")
PLAN = STRAT / "evaluation" / "folds" / "plan_cs_aro.json"
CONFIG = STRAT / "config" / "config_wf.json"

HORIZONS = (1, 2, 4, 8, 16, 32, 48, 96)
FEE = 0.0010  # binance.us 0.10 / 0.10 (memory reference_exchange_venues); config_wf has no fee key
SLIPPAGE_BPS = (0.0, 2.5, 5.0, 10.0)
VERDICT_BPS = 5.0
BARS_PER_DAY = 96
EARLY = tuple(f"f{i:02d}" for i in range(8))
LATE = tuple(f"f{i:02d}" for i in range(8, 15))
FWD_START = pd.Timestamp("2026-09-07", tz="UTC")


def pairs() -> list[str]:
    """The 10 pairs of config_wf.json (comments stripped)."""
    cfg = json.loads(re.sub(r"//.*", "", CONFIG.read_text()))
    return list(cfg["exchange"]["pair_whitelist"])


def folds() -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    """fold_id -> (test_start, test_end), end exclusive, from plan_cs_aro.json."""
    plan = json.loads(PLAN.read_text())
    return {f["fold_id"]: (pd.Timestamp(f["test_start"], tz="UTC"), pd.Timestamp(f["test_end"], tz="UTC"))
            for f in plan["folds"]}


def half_of(fold: str) -> str:
    return "early" if fold in EARLY else "late" if fold in LATE else "fwd" if fold == "fwd" else ""


def cost_rt(bps: float) -> float:
    """Round-trip cost as a log-return deduction: 2 x (fee + slippage per side)."""
    return 2.0 * (FEE + bps / 1e4)


def fname(pair: str) -> str:
    return pair.replace("/", "_")


def prereg() -> dict:
    return json.loads(PREREG.read_text())


def _git(*args) -> str:
    from research.control_sizer.judge import _git as g

    return g(*args)


def refusals(result_paths: list[Path] = (), prereg: Path | None = None) -> list[str]:
    """Why results may not be produced/read yet (FR-009): the pre-registration must be tracked,
    clean, and committed before every result file named here was written."""
    prereg = prereg or PREREG
    if not prereg.exists():
        return [f"{prereg} missing"]
    rel = str(prereg.relative_to(STRAT))
    if not _git("ls-files", "--", rel):
        return [f"{rel} is not committed"]
    if _git("status", "--porcelain", "--", rel):
        return [f"{rel} has uncommitted changes"]
    ts = float(_git("log", "-1", "--format=%ct", "--", rel) or 0)
    return [f"{p} is older than the pre-registration commit" for p in result_paths
            if Path(p).exists() and os.path.getmtime(p) < ts]


def refuse_unless_registered(result_paths: list[Path] = (), prereg: Path | None = None) -> int | None:
    """None when allowed; otherwise prints the reasons and returns exit code 2."""
    bad = refusals(result_paths, prereg)
    if bad:
        print("REFUSED:\n  " + "\n  ".join(bad))
        return 2
    return None


def write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    tmp.replace(path)


def write_feather(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.reset_index(drop=True).to_feather(tmp)
    tmp.replace(path)
