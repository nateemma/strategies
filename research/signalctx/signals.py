"""Offline reconstruction of EVERY entry signal, admitted or masked (T005, T006, T011a).

WHY THIS IS NOT THE SECOND-INFERENCE-PATH HAZARD. `research/labelfree/es/signals.py` deliberately
rejected offline INFERENCE, because this repository has a recorded case of predict-time scaling
silently diverging from production (the single-task GAN scaler bug, worth 1.53pp, found only by
hunting). That prohibition is about RUNNING THE MODEL a second way. This module runs the model
zero times: it consumes probability side-cars the production backtest path already produced and
recomputes only the GATE, which is arithmetic over recorded columns and is exactly what
`es/panel.py::entries()` already does.

A SIGNAL IS AN EVENT, NOT A BAR. Measured on f08: 139 bars carry `enter_long` but they form only
120 distinct events -- 14% are continuations of a signal already counted. Counting bars would
inflate every masked-signal statistic by that factor and double-count one opportunity that
happened to persist for three candles.

FIDELITY IS A GATE, NOT A DIAGNOSTIC. If recorded trades do not appear in the reconstruction,
the instrument is wrong and nothing downstream is evidence. f08 measures 70/71 = 0.986; the one
miss is the test range's FIRST bar, where rolling indicators lack history.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import FIDELITY_MIN, OUT, ROOT, SIDECAR_ROOT

sys.path.insert(0, str(ROOT))
from research.labelfree.es import panel as PA          # noqa: E402


def preserve() -> Path:
    """T001 -- copy the side-cars out of /tmp. NOT the 2.1GB panel cache: it rebuilds in 3.4s
    per fold, so preserving it would add 2.1GB to a repository to save under two minutes."""
    src = Path("/tmp/lf/es_sink")
    if not src.exists():
        raise SystemExit(f"{src} is gone -- regenerate with research.labelfree.es.signals (~44min)")
    SIDECAR_ROOT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, SIDECAR_ROOT, dirs_exist_ok=True)
    return SIDECAR_ROOT


def fold_params(fold_id: str) -> dict:
    """The fold's OWN params snapshot, never the live working file."""
    js = ROOT / "NNNC" / f"NNNC_WF_{fold_id}.json"
    buy = json.loads(js.read_text())["params"]["buy"]
    return {"entry_rvol_threshold": buy["entry_rvol_threshold"],
            "entry_guard_threshold": buy["entry_guard_threshold"]}


def _events(mask: np.ndarray) -> np.ndarray:
    """Indices where a contiguous run of True STARTS (R2)."""
    m = mask.astype(int)
    starts = np.flatnonzero(np.diff(np.r_[0, m]) == 1)
    return starts


def run_lengths(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(int)
    starts = _events(mask)
    ends = np.flatnonzero(np.diff(np.r_[m, 0]) == -1)
    return (ends - starts) + 1


def signal_bars(fold_id: str, train: bool = False) -> dict[str, set]:
    """EVERY bar carrying `enter_long`, not just run starts.

    Needed because a trade can open MID-RUN: when the pair was occupied at the run's first bar
    and freed part-way through, freqtrade enters on a continuation bar. Matching recorded trades
    against run STARTS alone reports those as misses -- measured 2026-09-16 on f04/f06/f09/f12/
    f13, where the previous bar carried a signal that was a continuation, dropping four folds
    below the fidelity floor for a reason that had nothing to do with the reconstruction.
    """
    p = fold_params(fold_id)
    d = PA.load(fold_id, train=train)
    out = {}
    for pair, dd in d["pairs"].items():
        mask = PA.entries(dd, p, enable_guards=True)
        out[pair] = set(pd.DatetimeIndex(dd["index"])[mask])
    return out


def reconstruct(fold_id: str, train: bool = False) -> pd.DataFrame:
    """Every entry signal on the fold's range, as EVENTS."""
    p = fold_params(fold_id)
    d = PA.load(fold_id, train=train)
    rows = []
    for pair, dd in d["pairs"].items():
        mask = PA.entries(dd, p, enable_guards=True)
        if not mask.any():
            continue
        idx = pd.DatetimeIndex(dd["index"])
        starts, lens = _events(mask), run_lengths(mask)
        for s, ln in zip(starts, lens):
            rows.append({"fold_id": fold_id, "pair": pair,
                         "timestamp": idx[s], "run_bars": int(ln)})
    return pd.DataFrame(rows).sort_values(["pair", "timestamp"]).reset_index(drop=True)


def recorded_trades(fold_id: str) -> pd.DataFrame:
    pool = json.loads((ROOT / "evaluation/pooled_nnnc.json").read_text())
    tr = [t for t in pool["trades"] if t["fold_id"] == fold_id]
    df = pd.DataFrame(tr)
    if len(df):
        df["open_date"] = pd.to_datetime(df["open_date"])
        df["close_date"] = pd.to_datetime(df["close_date"])
    return df


def fidelity(fold_id: str) -> dict:
    """Do recorded trades appear in the reconstruction? Below FIDELITY_MIN the fold is unusable.

    freqtrade opens on the candle AFTER the signal candle, so a trade matches if its open bar
    or the bar before it carries a signal.
    """
    sig = reconstruct(fold_id)
    tr = recorded_trades(fold_id)
    if not len(tr):
        return {"fold_id": fold_id, "n_trades": 0, "matched": 0, "rate": float("nan"),
                "usable": False, "misses": []}
    # Match against ALL signal bars, not run starts -- see signal_bars().
    by_pair = signal_bars(fold_id)
    matched, misses = 0, []
    for _, t in tr.iterrows():
        s = by_pair.get(t["pair"], set())
        if t["open_date"] in s or (t["open_date"] - pd.Timedelta(minutes=15)) in s:
            matched += 1
        else:
            misses.append({"pair": t["pair"], "open_date": str(t["open_date"])})
    rate = matched / len(tr)
    return {"fold_id": fold_id, "n_trades": int(len(tr)), "matched": int(matched),
            "rate": float(rate), "usable": bool(rate >= FIDELITY_MIN),
            "n_signal_events": int(len(sig)),
            "n_signal_bars": int(sig["run_bars"].sum()) if len(sig) else 0,
            "misses": misses[:5]}


def census(fold_ids: list[str]) -> dict:
    """T011a -- the all-fold gate. f08 alone is not evidence about the other fourteen."""
    rows = [fidelity(f) for f in fold_ids]
    usable = [r for r in rows if r["usable"]]
    rates = [r["rate"] for r in rows if r["rate"] == r["rate"]]
    return {"folds": rows, "n_folds": len(rows), "n_usable": len(usable),
            "min_rate": float(min(rates)) if rates else float("nan"),
            "mean_rate": float(np.mean(rates)) if rates else float("nan"),
            "gate": "PASS" if len(usable) == len(rows) else "FAIL",
            "fidelity_min": FIDELITY_MIN}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preserve", action="store_true")
    ap.add_argument("--census", action="store_true")
    ap.add_argument("--fold", default="all")
    a = ap.parse_args()
    if a.preserve:
        print("preserved ->", preserve()); return 0

    sys.path.insert(0, str(ROOT))
    from research.labelfree import folds as FO
    ids = [f.fold_id for f in FO.folds()] if a.fold == "all" else a.fold.split(",")

    if a.census:
        c = census(ids)
        print(f"{'fold':>5} {'trades':>7} {'matched':>8} {'rate':>7} {'events':>7} {'bars':>6} {'usable':>7}")
        for r in c["folds"]:
            print(f"{r['fold_id']:>5} {r['n_trades']:>7} {r['matched']:>8} {r['rate']:>7.3f} "
                  f"{r['n_signal_events']:>7} {r['n_signal_bars']:>6} {str(r['usable']):>7}")
        print(f"\nGATE {c['gate']}: {c['n_usable']}/{c['n_folds']} folds usable "
              f"(min {c['min_rate']:.3f}, mean {c['mean_rate']:.3f}, floor {FIDELITY_MIN})")
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "fidelity.json").write_text(json.dumps(c, indent=2))
        print(f"WROTE {OUT/'fidelity.json'}")
        return 0 if c["gate"] == "PASS" else 1
    for f in ids:
        print(f, len(reconstruct(f)), "signal events")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
