"""End-to-end trigger causality, measured by truncation (feature 023, research R11, FR-003).

The registry (`triggers.py`) blocks outcome and pair columns structurally, but it cannot see a
trigger -- or a FEATURE a trigger reads -- that looks forward. `triggers.py:239-257` records that
pandas-swallowed warm-up NaNs even evade its NaN guard. So causality is measured, not argued:

    compute the production feature frame + the trigger on a FULL slice of history, and again on
    PREFIXES of that slice cut at K points; over every prefix (past a declared warm-up) the two
    trigger outputs must be IDENTICAL.

Any difference means the value at some bar depended on bars after it. The feature frame is built
by `research.clfstudy.data.populate` -- the same `DataframePopulator` MINIMAL call the strategies
make -- so a non-causal INDICATOR is caught here too, not only a non-causal trigger.

A PLANTED CONTROL must FAIL, or the checker itself is broken (constitution IV: a probe that
cannot fail reports the same "0" as a real null). The control is `local_min_followthrough`'s
entry condition, `is_local_min` over a CENTRED window, which reads k bars ahead (021 audit).

Verdicts are written to `trigger_causality.json`, each with the git blob sha of the code that
produced it. `check_runnable()` -- used by SimpleTrigger strategies and by `run_folds`' no-model
preflight -- accepts a trigger only when its row is PASS AND that code is unchanged; a stale row
counts as not PASS.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd


STRAT_DIR = Path(__file__).resolve().parents[2]
VERDICT_FILE = Path(__file__).resolve().parent / "trigger_causality.json"
CODE_FILES = (
    "utils/DataframePopulator.py",
    "research/outcomes/builtin_triggers.py",
    "research/outcomes/audit_triggers.py",
    "research/outcomes/triggers.py",
    "research/outcomes/causality.py",
)

#: Whitelist pairs whose 15m history is CONTIGUOUS through all three eras. XRP is not: it was
#: delisted from binance.us 2021-2023, so a "2021" XRP slice silently spans a 2.5-year gap.
#: LINK/NEAR/ZEC/AVAX/DOT start after the first era. BCH is contiguous but heavily gap-filled.
DEFAULT_PAIRS = ("SOL", "LTC", "AAVE")
#: three non-adjacent eras (era start, UTC)
DEFAULT_ERAS = ("2021-10-01", "2023-10-01", "2025-07-01")
HISTORY_BARS = 1000  # history before the era start, so long indicators are warm
MAX_GAP = pd.Timedelta("1D")  # a slice with a larger hole is not an era
ERA_BARS = 6000  # ~62 days of 15m bars per era
#: Detection is per cut: a trigger reading k bars ahead is caught when its output near a cut
#: depends on the bars beyond it. Measured 2026-10-10: the planted k=3 control FAILS on 9/9
#: full slices at 8 cuts; 16 is the default for margin.
DEFAULT_CUTS = 16
DEFAULT_WARMUP = 200  # = SimpleTrigger startup_candle_count; excluded from comparison

CONTROL_NAME = "local_min_followthrough"
CONTROL_K = 3


def _ensure_paths() -> None:
    for p in (str(STRAT_DIR), str(STRAT_DIR / "utils")):
        if p not in sys.path:
            sys.path.insert(0, p)


def code_sha() -> str:
    """git blob shas of every file that can change a verdict, hashed together."""
    shas = []
    for rel in CODE_FILES:
        out = subprocess.run(
            ["git", "hash-object", rel], cwd=STRAT_DIR, capture_output=True, text=True, check=True
        )
        shas.append(out.stdout.strip())
    import hashlib

    return hashlib.sha256("".join(shas).encode()).hexdigest()[:16]


def control_trigger():
    """The planted forward-reading condition: a centred-window local minimum."""
    _ensure_paths()
    from research.outcomes.triggers import Trigger

    def _centred_min(df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        m = close.rolling(2 * CONTROL_K + 1, center=True, min_periods=1).min()
        return (close <= m).fillna(False).astype(bool)

    return Trigger(
        name=CONTROL_NAME,
        evaluate=_centred_min,
        required_columns=("close",),
        source="CONTROL",
        description="centred local minimum -- reads k bars ahead",
    )


def _populate(ohlcv: pd.DataFrame) -> pd.DataFrame:
    _ensure_paths()
    from research.clfstudy.data import populate

    return populate(ohlcv.reset_index())


def _load_slice(pair: str, era_start: str) -> pd.DataFrame | None:
    _ensure_paths()
    from regime.regimedata import load_ohlcv

    ohlcv = load_ohlcv(pair)
    start = pd.Timestamp(era_start, tz="UTC")
    i0 = ohlcv.index.searchsorted(start)
    lo = max(0, i0 - HISTORY_BARS)
    hi = i0 + ERA_BARS
    if hi > len(ohlcv) or i0 - lo < HISTORY_BARS // 2:
        return None
    sl = ohlcv.iloc[lo:hi]
    # Contiguity: searchsorted lands on the first bar AFTER a listing gap, so without this a
    # requested era can silently become a slice spanning years (found on XRP, 2026-10-10).
    if sl.index.to_series().diff().max() > MAX_GAP or abs(sl.index[HISTORY_BARS] - start) > MAX_GAP:
        return None
    return sl


def _evaluate(trigger, frame: pd.DataFrame) -> np.ndarray | None:
    """Trigger output as a bool array; None when its columns are absent (UNSCOREABLE)."""
    from research.outcomes import triggers as T

    missing = [c for c in trigger.required_columns if c not in frame.columns]
    if missing:
        return None
    return T.evaluate(trigger, frame).to_numpy(dtype=bool)


def check_triggers(
    trigger_list,
    pairs=DEFAULT_PAIRS,
    eras=DEFAULT_ERAS,
    cuts: int = DEFAULT_CUTS,
    warmup: int = DEFAULT_WARMUP,
    slice_loader=None,
) -> dict[str, dict]:
    """Truncation test for every trigger in `trigger_list`. Feature frames are shared."""
    load = slice_loader or _load_slice
    res = {
        t.name: {
            "trigger": t.name,
            "verdict": "PASS",
            "first_mismatch": None,
            "comparisons": 0,
            "missing_columns": None,
        }
        for t in trigger_list
    }
    slices = {"checked": [], "skipped": []}
    for pair in pairs:
        for era in eras:
            ohlcv = load(pair, era)
            if ohlcv is None:
                slices["skipped"].append(f"{pair}@{era}")
                continue
            slices["checked"].append(f"{pair}@{era}")
            full = _populate(ohlcv)
            n = len(full)
            full_out = {t.name: _evaluate(t, full) for t in trigger_list}
            cut_points = np.linspace(warmup + 50, n - 1, cuts).astype(int)
            for cut in cut_points:
                prefix = _populate(ohlcv.iloc[: cut + 1])
                for t in trigger_list:
                    r = res[t.name]
                    if full_out[t.name] is None:
                        r["verdict"] = "UNSCOREABLE"
                        r["missing_columns"] = [
                            c for c in t.required_columns if c not in full.columns
                        ]
                        continue
                    if r["verdict"] != "PASS":
                        continue
                    a = full_out[t.name][warmup : cut + 1]
                    b = _evaluate(t, prefix)[warmup : cut + 1]
                    r["comparisons"] += 1
                    if not np.array_equal(a, b):
                        j = int(np.flatnonzero(a != b)[0]) + warmup
                        r["verdict"] = "FAIL"
                        r["first_mismatch"] = {
                            "pair": pair,
                            "era": era,
                            "cut_bar": int(cut),
                            "bar": j,
                            "bars_before_cut": int(cut - j),
                            "date": str(ohlcv.index[j]),
                            "full": bool(full_out[t.name][j]),
                            "truncated": bool(_evaluate(t, prefix)[j]),
                        }
    for r in res.values():
        if r["verdict"] == "PASS" and r["comparisons"] == 0:
            r["verdict"] = "UNSCOREABLE"  # nothing was compared: never a PASS
    res["__slices__"] = slices
    return res


def run(pairs=DEFAULT_PAIRS, eras=DEFAULT_ERAS, cuts=DEFAULT_CUTS, names=None) -> dict:
    _ensure_paths()
    from research.outcomes import triggers as T

    T.ensure_builtins()
    tlist = [T.get(n) for n in (names or T.names())]
    control = control_trigger()
    results = check_triggers(tlist + [control], pairs, eras, cuts)
    ctl = results.pop(CONTROL_NAME)
    slices = results.pop("__slices__")
    return {
        "generated": datetime.now(UTC).isoformat(timespec="seconds"),
        "code_sha": code_sha(),
        "pairs": list(pairs),
        "eras": list(eras),
        "cut_points": cuts,
        "warmup_bars": DEFAULT_WARMUP,
        "history_bars": HISTORY_BARS,
        "era_bars": ERA_BARS,
        "slices_checked": slices["checked"],
        "slices_skipped": slices["skipped"],
        "checker_control": {
            "trigger": CONTROL_NAME,
            "verdict": ctl["verdict"],
            "first_mismatch": ctl["first_mismatch"],
        },
        "triggers": results,
    }


def check_runnable(name: str | None, verdict_file: Path | None = None) -> tuple[bool, str]:
    """(ok, reason): may a SimpleTrigger arm trade this trigger? Fails closed."""
    if not name:
        return False, "no TRIGGER_NAME"
    vf = verdict_file or VERDICT_FILE
    if not vf.is_file():
        return (
            False,
            f"no causality verdicts at {vf} -- run `python -m research.outcomes.causality --write`",
        )
    try:
        data = json.loads(vf.read_text())
    except ValueError as exc:
        return False, f"unparseable verdict file ({exc})"
    if data.get("checker_control", {}).get("verdict") != "FAIL":
        return False, "verdict file is invalid: the planted control did not FAIL"
    row = data.get("triggers", {}).get(name)
    if row is None:
        return False, f"{name!r} has no causality verdict"
    if row.get("verdict") != "PASS":
        return False, f"{name!r} causality verdict is {row.get('verdict')}"
    if data.get("code_sha") != code_sha():
        return False, (
            "causality verdicts are STALE: populator/trigger/checker code changed "
            "since they were measured -- re-run the checker"
        )
    return True, "PASS"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trigger", action="append", help="limit to these triggers (repeatable)")
    ap.add_argument("--pairs", nargs="+", default=list(DEFAULT_PAIRS))
    ap.add_argument("--cuts", type=int, default=DEFAULT_CUTS)
    ap.add_argument("--write", action="store_true", help=f"write {VERDICT_FILE.name}")
    a = ap.parse_args(argv)
    out = run(pairs=tuple(a.pairs), cuts=a.cuts, names=a.trigger)
    ctl = out["checker_control"]
    print(f"slices checked {len(out['slices_checked'])}: {', '.join(out['slices_checked'])}")
    if out["slices_skipped"]:
        print(
            f"*** slices SKIPPED (not contiguous / too short): {', '.join(out['slices_skipped'])}"
        )
    print(f"control {ctl['trigger']}: {ctl['verdict']}  (must be FAIL)")
    for name, r in sorted(out["triggers"].items()):
        extra = ""
        if r["verdict"] == "FAIL":
            fm = r["first_mismatch"]
            extra = (
                f"  first mismatch {fm['pair']} {fm['date']} "
                f"({fm['bars_before_cut']} bars before cut)"
            )
        elif r["verdict"] == "UNSCOREABLE" and r["missing_columns"]:
            extra = f"  missing {r['missing_columns']}"
        print(f"  {name:<20} {r['verdict']:<12} comparisons={r['comparisons']}{extra}")
    if ctl["verdict"] != "FAIL":
        print("*** CHECKER BROKEN: the planted forward-reading control did not FAIL")
        return 1
    if a.write:
        VERDICT_FILE.write_text(json.dumps(out, indent=2) + "\n")
        print(f"written: {VERDICT_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
