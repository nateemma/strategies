"""034 S0a: label audit before training (FR-008a, research R1/R2/R10). Labels only -- no model is
loaded or trained.

Per candidate and early fold, on the fold's TRAINING window, through each candidate class's own
label path: the indicator steps BaseNNStrategy.populate_indicators runs before labelling
(precision columns, MINIMAL indicators, additional and debug indicators), then
`get_train_buy_signals` / `get_train_sell_signals` (raw, after the peak filter) and
`get_training_labels` (after conflict resolution: the classes the model is trained on).

Recorded: BUY / SELL / HOLD shares of kept classes, the raw BUY-SELL overlap, Spearman(label,
bb_width) on rule-eligible bars (guard_metric < 0.3 AND bb_width > 0.035) for BUY and SELL, and
whether the indicator columns equal the control's. On f00 only, per declared extra, whether it
changes the raw labels when removed: an extra equal to its labeler's default is `equal_default`;
one that differs but changes nothing is `no_op` (an S0a failure).

Status INCONCLUSIVE if the kept BUY share is under FLOOR in any early fold, or any extra is no_op.
CONTROL and GFM are reference rows: recorded, never gated.

    PYTHONPATH=. ../../.venv/bin/python -m research.label_screen.audit --fold f00   # one per process
    PYTHONPATH=. ../../.venv/bin/python -m research.label_screen.audit --merge
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

import numpy as np


STRAT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(STRAT), str(STRAT / "evaluation")]

from research.label_screen import candidates as C  # noqa: E402


HERE = Path(__file__).resolve().parent
OUT = HERE / "audit"
FOLDS = STRAT / "evaluation" / "folds"
CONFIG = "user_data/strategies/config/config_wf.json"
FLOOR = 0.01
EARLY = tuple(f"f{i:02d}" for i in range(8))
from Framework.BaseStrategy import TradingAction  # noqa: E402

BUY, SELL = int(TradingAction.BUY), int(TradingAction.SELL)  # the trained classes (HOLD between)

#: the forward_mae sell branch is inline in get_train_sell_signals (no function to inspect)
_INLINE_SELL_DEFAULTS = {0: {"horizon": 72, "min_loss": 0.02, "max_drawdown": 0.02}}


def class_for(c: C.Candidate) -> str:
    return "NNNC_A_Raw_Open" if c.tag == "CONTROL" else C.class_name(c)


def labeler_default(method: int, key: str, side: str):
    from Framework import TrainingSignals as T

    if side == "sell" and method in _INLINE_SELL_DEFAULTS:
        return _INLINE_SELL_DEFAULTS[method].get(key, inspect.Parameter.empty)
    fn = T.METHODS[T.LabelMethod(method)]
    if side == "sell":
        fn = getattr(T, f"{fn.__name__}_sell", fn)
    p = inspect.signature(fn).parameters.get(key)
    return inspect.Parameter.empty if p is None else p.default


def liveness_status(value, default, changed: int) -> str:
    if changed > 0:
        return "live"
    return "equal_default" if value == default else "no_op"


def shares(classes: np.ndarray) -> dict:
    n = max(len(classes), 1)
    return {"buy": float(np.sum(classes == BUY) / n), "sell": float(np.sum(classes == SELL) / n),
            "hold": float(np.sum((classes != BUY) & (classes != SELL)) / n), "rows": int(len(classes))}


def spearman(label: np.ndarray, width: np.ndarray) -> float | None:
    from scipy.stats import spearmanr

    if len(label) < 3 or np.all(label == label[0]):
        return None
    return float(spearmanr(label, width).statistic)


def status(rows: dict) -> tuple[str, str]:
    """rows: {fold: audit row}. PASS or INCONCLUSIVE with a reason."""
    low = sorted(f for f, r in rows.items() if r["kept"]["buy"] < FLOOR)
    if low:
        return "INCONCLUSIVE", f"kept BUY share < {FLOOR} in {low}"
    noop = sorted({k for r in rows.values() for k, v in (r.get("liveness") or {}).items()
                   if v["status"] == "no_op"})
    if noop:
        return "INCONCLUSIVE", f"declared extra(s) with no effect: {noop}"
    return "PASS", ""


def _frames(cls_name: str, fid: str):
    import pick_threshold as P
    from freqtrade.data.converter import trim_dataframe
    from utils.DataframePopulator import DatasetType

    plan = json.loads((FOLDS / "plan_wf_aro_g24.json").read_text())
    f = next(x for x in plan["folds"] if x["fold_id"] == fid)
    strat, cfg = P._load_fold_context(cls_name, "NNNC", CONFIG)
    data, timerange = P._load_train_data(strat, cfg, P.TrainWindow.from_fold(f))
    for pair, df in data.items():
        if df.empty:
            continue
        strat.curr_pair = pair
        strat.iteration_init()
        d = strat.check_precision_columns(df.copy())
        d = strat.dataframePopulator.add_indicators(d, dataset_type=DatasetType.MINIMAL)
        d = strat.add_additional_indicators(d)
        d = strat.add_debug_indicators(d)
        yield pair, strat, trim_dataframe(d, timerange, startup_candles=strat.startup_candle_count)


def audit_one(c: C.Candidate, fid: str, control_cols: list | None) -> tuple[dict, list]:
    from Framework import TrainingSignals as T

    kept, rawb, raws, widths, guards, cols = [], [], [], [], [], None
    live = {} if fid == "f00" and c.tag not in ("CONTROL", "GFM") else None
    changed = {}
    for pair, strat, d in _frames(class_for(c), fid):
        cols = list(d.columns)
        b = np.asarray(strat.get_train_buy_signals(d.copy()), dtype=float)
        s = np.asarray(strat.get_train_sell_signals(d.copy()), dtype=float)
        k = np.asarray(strat.get_training_labels(d.copy()))
        n = len(d)
        rawb.append(b[-n:] > 0.5)
        raws.append(s[-n:] > 0.5)
        kept.append(k[-n:])
        widths.append(d["bb_width"].to_numpy(float))
        guards.append(d["guard_metric"].to_numpy(float))
        if live is not None:
            for side, extra in (("buy", c.label_params), ("sell", C.sell_params(c))):
                for key in extra:
                    full = strat._label_params(side)
                    less = {x: v for x, v in full.items() if x != key}
                    fn = T.get_train_buy_signals if side == "buy" else T.get_train_sell_signals
                    a = np.asarray(fn(d.copy(), method=c.method, params=full), dtype=float)
                    z = np.asarray(fn(d.copy(), method=c.method, params=less), dtype=float)
                    changed[(side, key)] = changed.get((side, key), 0) + int(np.sum(a != z))
    kept, rawb, raws = np.concatenate(kept), np.concatenate(rawb), np.concatenate(raws)
    width, guard = np.concatenate(widths), np.concatenate(guards)
    elig = (guard < 0.3) & (width > 0.035)
    row = {"kept": shares(kept),
           "raw": {"buy": float(rawb.mean()), "sell": float(raws.mean()),
                   "overlap": float((rawb & raws).mean()),
                   "overlap_of_buy": float((rawb & raws).sum() / max(rawb.sum(), 1))},
           "eligible_rows": int(elig.sum()),
           "spearman_width_eligible": {
               "buy": spearman((kept[elig] == BUY).astype(float), width[elig]),
               "sell": spearman((kept[elig] == SELL).astype(float), width[elig])},
           "columns_equal_control": None if control_cols is None else cols == control_cols}
    if live is not None:
        for (side, key), n_changed in changed.items():
            extra = c.label_params if side == "buy" else C.sell_params(c)
            d0 = labeler_default(c.method, key, side)
            live[f"{side}.{key}"] = {"value": extra[key], "default": None if d0 is inspect.Parameter.empty else d0,
                                     "changed": n_changed,
                                     "status": liveness_status(extra[key], d0, n_changed)}
        row["liveness"] = live
    return row, cols


def run_fold(fid: str, tags: set | None = None) -> dict:
    """All candidates, or only `tags` (merged into the fold's existing file; CONTROL is always
    recomputed as the column reference)."""
    OUT.mkdir(exist_ok=True)
    path = OUT / f"{fid}.json"
    out = json.loads(path.read_text()) if tags and path.is_file() else {}
    control_cols = None
    for c in C.CANDIDATES:  # CONTROL first: its columns are the reference
        if tags and c.tag != "CONTROL" and c.tag not in tags:
            continue
        row, cols = audit_one(c, fid, control_cols)
        if c.tag == "CONTROL":
            control_cols = cols
            row["columns_equal_control"] = True
        out[c.tag] = row
        print(fid, c.tag, json.dumps({"kept": row["kept"], "overlap": row["raw"]["overlap"],
                                      "cols": row["columns_equal_control"],
                                      "liveness": row.get("liveness")}), flush=True)
    path.write_text(json.dumps(out, indent=1))
    return out


def merge() -> dict:
    per = {f: json.loads((OUT / f"{f}.json").read_text()) for f in EARLY if (OUT / f"{f}.json").is_file()}
    missing = [f for f in EARLY if f not in per]
    res = {"label": "034 S0a label audit (labels only; no model)", "floor": FLOOR,
           "folds": sorted(per), "folds_missing": missing, "candidates": {}}
    for c in C.CANDIDATES:
        rows = {f: per[f][c.tag] for f in per}
        st, why = status(rows) if not missing else ("INCONCLUSIVE", f"folds missing {missing}")
        if not all(r["columns_equal_control"] for r in rows.values()):
            st, why = "INCONCLUSIVE", "indicator columns differ from the control's"
        res["candidates"][c.tag] = {"status": "REFERENCE" if c.tag in ("CONTROL", "GFM") else st,
                                    "reason": why, "folds": rows}
    (HERE / "audit.json").write_text(json.dumps(res, indent=1))
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--fold", choices=EARLY)
    g.add_argument("--merge", action="store_true")
    ap.add_argument("--tags", help="comma-separated tags to recompute in an existing fold file")
    a = ap.parse_args(argv)
    if a.merge:
        r = merge()
        for t, v in r["candidates"].items():
            print(t, v["status"], v["reason"])
    else:
        run_fold(a.fold, set(a.tags.split(",")) if a.tags else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
