"""Per-fold, leakage-free `prediction_threshold` selection (walk-forward harness).

WHY THIS EXISTS. `prediction_threshold` is a property of a model's own predicted-probability
distribution, not a portable constant. 0.80 was tuned for a gbb-labelled model, which is
near-deterministic from its own trigger and therefore very confident; a model trained on a
genuinely noisy target (e.g. forward_mae) sits near the class priors and may never clear 0.80,
giving it zero trades for a reason that has nothing to do with whether its signal is any good.
This script picks the threshold that reproduces a TARGET entry rate on a model's own training
data, so an arm and a control can be compared at matched operating points instead of a borrowed
constant.

LEAKAGE IS THE WHOLE POINT. Every data-loading path below is typed to accept only a
`TrainWindow`, constructed solely by `TrainWindow.from_fold`, which asserts the window ends at
or before the fold's test window starts. There is no function in this file that accepts a bare
timerange string or the fold dict itself for data loading, so there is no code path that could
reach for `fold["test_timerange"]` by accident.

NOT A BACKTEST. No `Backtesting`, `Wallets`, or trade-simulation machinery is constructed. This
loads OHLCV data, runs the fold strategy's OWN `populate_indicators` / `populate_entry_trend`
(via freqtrade's own `advise_all_indicators` / `advise_entry` -- never a reimplementation of
feature-building) and reads the `%predict_prob_*` columns `get_predictions` already writes.
That is a single forward pass through an already-trained model; it never trains one. A fold
with no trained model yet (`run_folds.fold_has_model` returns False) is SKIPPED, not trained.

HOW A CHOSEN THRESHOLD REACHES THE FOLD RUN. See `run_folds.py`'s optional `thresholds` plan
key and `generate_fold`'s `threshold` argument: the chosen value is written into the fold's
GENERATED `.json` (still derived from the frozen params snapshot, never the live working file),
not applied by this script directly -- this script only measures.

Usage (from the repo root, with the harness's venv):

    PYTHONPATH=user_data/strategies .venv/bin/python user_data/strategies/evaluation/pick_threshold.py \\
        --plan user_data/strategies/evaluation/folds/plan_NNNC_ForwardMAE_Raw.json \\
        --fold f00 --target-rate 0.05

    # every fold that already has a trained model, written to a thresholds map:
    .venv/bin/python user_data/strategies/evaluation/pick_threshold.py \\
        --plan ...plan_NNNC_ForwardMAE_Raw.json --target-rate 0.05 \\
        --out user_data/strategies/evaluation/folds/thresholds_fmr.json

    # what rate does the CONTROL's own pinned threshold achieve on its own train windows?
    .venv/bin/python user_data/strategies/evaluation/pick_threshold.py \\
        --plan ...plan_NNNC_A_Raw_Open.json --mode control-rate \\
        --out user_data/strategies/evaluation/folds/control_rates_aro.json
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# parents: [0]=evaluation [1]=strategies [2]=user_data [3]=repo root. Same indexing as
# run_folds.py (fixed there 2026-10-07) -- this file lives in the same directory.
ROOT = Path(__file__).resolve().parents[3]
STRAT_DIR = ROOT / "user_data/strategies"

sys.path.insert(0, str(Path(__file__).resolve().parent))   # import run_folds regardless of cwd
sys.path.insert(0, str(STRAT_DIR))                          # Framework/ as a top-level package

import run_folds  # noqa: E402  -- fold_has_model: the "is this fold trained" check (030 model key)
from Framework.BaseStrategy import TradingAction  # noqa: E402

# Column order matches TradingAction's int values (SELL=0, HOLD=1, BUY=2): get_predictions
# (Framework/BaseNNStrategy.py) writes these straight from pred_probs[:, TradingAction.X], so
# stacking them in THIS order and calling np.argmax reproduces argmax_with_threshold's own
# tie-break (lowest index wins) on the identical underlying array.
PROB_COLS = ("%predict_prob_sell", "%predict_prob_hold", "%predict_prob_buy")


# ---------------------------------------------------------------------------
# Leakage guard: the only sanctioned way to get a timerange out of a fold
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainWindow:
    """A fold's TRAIN-only timerange -- the only timerange any data-loading function in this
    file is allowed to take. Construct only via `from_fold`; there is deliberately no other
    constructor, so a caller cannot hand this script `fold["test_timerange"]` by mistake.
    """

    fold_id: str
    timerange: str

    @classmethod
    def from_fold(cls, fold: dict) -> "TrainWindow":
        train_tr, test_tr = fold["train_timerange"], fold["test_timerange"]
        train_end = train_tr.split("-")[1]
        test_start = test_tr.split("-")[0]
        assert train_end <= test_start, (
            f"fold {fold['fold_id']}: train_timerange {train_tr} extends past the start of "
            f"test_timerange {test_tr} -- refusing a window that could leak test data into "
            f"threshold selection"
        )
        return cls(fold_id=fold["fold_id"], timerange=train_tr)


# ---------------------------------------------------------------------------
# Pure threshold math -- no freqtrade, no I/O; the part the tests hit hardest
# ---------------------------------------------------------------------------


def effective_buy_probs(df: pd.DataFrame) -> np.ndarray:
    """Per-bar BUY probability where BUY is the argmax class, 0.0 everywhere else.

    Mirrors `BaseNNStrategy.argmax_with_threshold` exactly: a bar only ever becomes a BUY
    trading decision if BUY both has the highest probability AND clears the threshold.
    Un-scored bars (NaN in any of the three probability columns -- the padded warm-up region
    `get_predictions` marks with NaN, never 0.0) are dropped, not treated as "not BUY".
    """
    probs = df[list(PROB_COLS)].to_numpy(dtype=float)
    scored = ~np.isnan(probs).any(axis=1)
    probs = probs[scored]
    if len(probs) == 0:
        return np.empty(0, dtype=float)
    is_buy_argmax = np.argmax(probs, axis=1) == TradingAction.BUY
    return np.where(is_buy_argmax, probs[:, TradingAction.BUY], 0.0)


def rate_for_threshold(effective_probs: np.ndarray, threshold: float) -> float:
    """Fraction of scored bars that would be a BUY decision at `threshold`."""
    if len(effective_probs) == 0:
        return 0.0
    return float(np.mean(effective_probs > threshold))


def threshold_for_rate(effective_probs: np.ndarray, target_rate: float) -> tuple[float, float]:
    """(threshold, achieved_rate): the threshold whose BUY rate on `effective_probs` is as
    close to `target_rate` as the data allows, and the rate actually achieved at it.

    `rate_for_threshold` is a survival function (monotonically non-increasing in `threshold`),
    so its inverse is the empirical quantile at `1 - target_rate`. The achievable rate is
    capped by the fraction of bars where BUY is the argmax at all: bars pinned at the 0.0
    sentinel can never cross a threshold > 0, so a `target_rate` above that cap returns
    threshold 0.0 and an `achieved` rate below the request -- always reported honestly rather
    than silently clipped.
    """
    n = len(effective_probs)
    if n == 0:
        raise ValueError("no scored bars to pick a threshold from")
    if not 0.0 <= target_rate <= 1.0:
        raise ValueError(f"target_rate must be in [0, 1], got {target_rate}")
    if target_rate <= 0.0:
        threshold = 1.0           # unreachable by a strict '>' probability compare -> rate 0
    else:
        threshold = float(np.quantile(effective_probs, 1.0 - target_rate))
    achieved = rate_for_threshold(effective_probs, threshold)
    return threshold, achieved


# ---------------------------------------------------------------------------
# freqtrade integration -- load the fold EXACTLY as a real backtest would
# ---------------------------------------------------------------------------


def _load_fold_context(cls: str, family: str, config: str):
    """Load the fold strategy through `StrategyResolver` -- never a reimplementation of its
    feature building -- and wire the minimal `DataProvider` its `populate_indicators` /
    `populate_entry_trend` need. No `Backtesting`, `Wallets`, or trade machinery: this is
    feature + prediction computation only.
    """
    from freqtrade.configuration import Configuration
    from freqtrade.data.dataprovider import DataProvider
    from freqtrade.enums import RunMode
    from freqtrade.resolvers import StrategyResolver

    cfg_path = Path(config)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    args = {
        "config": [str(cfg_path)],
        "strategy": cls,
        "user_data_dir": ROOT / "user_data",
        "strategy_path": str(STRAT_DIR / family),
    }
    cfg = Configuration(args, RunMode.BACKTEST).get_config()
    strat = StrategyResolver.load_strategy(cfg)
    strat.dbg_verbose = False             # see scripts/check_label_fidelity.py precedent
    strat.dp = DataProvider(cfg, None)    # exchange=None: all data comes from history.load_data

    # TrainingEngine.maybe_train's multi-pair AGGREGATE branch calls self.dp.current_whitelist(),
    # which needs a real PairListManager this script never constructs (no exchange, no live
    # pairlist resolution -- out of scope for feature computation). Forcing it off is safe here
    # specifically because a) this script NEVER trains (callers only reach this point after
    # run_folds.fold_has_model() is True, and the strategy's own model_exists() gate inside
    # maybe_train would skip training anyway), and b) the aggregate branch's only dataframe
    # transform, add_sequential_index, is presently a documented no-op
    # (Framework/BaseNNStrategy.py: "seq_index removed as per USER request"), so disabling
    # aggregation changes no computed value, only how prediction reaches that computation.
    # This mirrors what BaseNNStrategy.populate_indicators itself does for every non-backtest
    # runmode (dry-run/live) -- it is not a special case invented for this script.
    strat.aggregate_pairs = False
    strat.training_needed = False         # belt-and-suspenders on top of model_exists()
    return strat, cfg


def _load_train_data(strat, cfg, window: TrainWindow):
    from freqtrade.configuration import TimeRange
    from freqtrade.data import history

    timerange = TimeRange.parse_timerange(window.timerange)
    pairs = cfg["exchange"]["pair_whitelist"]
    data = history.load_data(
        datadir=cfg["datadir"],
        pairs=pairs,
        timeframe=cfg["timeframe"],
        timerange=timerange,
        startup_candles=strat.startup_candle_count,
        fail_without_data=True,
        data_format=cfg.get("dataformat_ohlcv", "feather"),
        candle_type=cfg.get("candle_type_def", "spot"),
    )
    return data, timerange


def predicted_buy_probs(strat, cfg, window: TrainWindow) -> np.ndarray:
    """Pooled-across-pairs effective BUY probability for `window`'s TRAIN range, computed
    through the strategy's own `populate_indicators` -> `populate_entry_trend` path -- the
    same calls freqtrade's own backtester makes per pair
    (freqtrade/optimize/backtesting.py:_get_ohlcv_as_lists). The fold's model must already
    exist; this function never trains one (callers check `run_folds.fold_has_model` first).
    """
    from freqtrade.data.converter import trim_dataframe

    data, timerange = _load_train_data(strat, cfg, window)
    preprocessed = strat.advise_all_indicators(data)   # builds/loads self.classifier lazily;
                                                        # does NOT retrain, the model exists
    pooled = []
    for pair, df in preprocessed.items():
        if df.empty:
            continue
        signalled = strat.advise_entry(df.copy(), {"pair": pair})
        trimmed = trim_dataframe(signalled, timerange, startup_candles=strat.startup_candle_count)
        pooled.append(effective_buy_probs(trimmed))
    if not pooled:
        raise RuntimeError(f"fold {window.fold_id}: no pairs produced any scored bars")
    return np.concatenate(pooled)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _trained(plan: dict, fold: dict) -> bool:
    """Does the fold's MODEL KEY have weights? Feature 030 stores weights at the key's location,
    not saved_data/<class>/, so the old class-named check skipped every keyed fold (032 R6)."""
    return run_folds.fold_has_model(fold, plan["base_strategy"], plan["config"], 1)


def _select_folds(plan: dict, fold_id: str | None) -> list[dict]:
    folds = plan["folds"]
    if fold_id is None:
        return folds
    out = [f for f in folds if f["fold_id"] == fold_id]
    if not out:
        raise SystemExit(f"no fold {fold_id!r} in plan (have: {[f['fold_id'] for f in folds]})")
    return out


def cmd_pick(plan: dict, fold_id: str | None, target_rate: float) -> dict:
    family, config = plan["family"], plan["config"]
    out: dict[str, float] = {}
    for f in _select_folds(plan, fold_id):
        cls = f["strategy_class"]
        if not _trained(plan, f):
            print(f"{f['fold_id']}: SKIP -- no trained model for {cls} (model key) yet")
            continue
        strat, cfg = _load_fold_context(cls, family, config)
        window = TrainWindow.from_fold(f)
        probs = predicted_buy_probs(strat, cfg, window)
        threshold, achieved = threshold_for_rate(probs, target_rate)
        print(f"{f['fold_id']}: threshold={threshold:.4f}  achieved_rate={achieved:.4%}  "
              f"target_rate={target_rate:.4%}  n_scored={len(probs)}")
        out[f["fold_id"]] = threshold
    return out


def cmd_control_rate(plan: dict, fold_id: str | None, threshold: float | None) -> dict:
    family, config = plan["family"], plan["config"]
    if threshold is None:
        snap_path = Path(plan["params_snapshot"])
        if not snap_path.is_absolute() and not snap_path.exists():
            snap_path = ROOT / snap_path
        snap = json.loads(snap_path.read_text())
        threshold = snap["params"]["buy"]["prediction_threshold"]
        print(f"(--threshold not given; read {threshold} from {plan['params_snapshot']})")
    out: dict[str, float] = {}
    for f in _select_folds(plan, fold_id):
        cls = f["strategy_class"]
        if not _trained(plan, f):
            print(f"{f['fold_id']}: SKIP -- no trained model for {cls} (model key) yet")
            continue
        strat, cfg = _load_fold_context(cls, family, config)
        window = TrainWindow.from_fold(f)
        probs = predicted_buy_probs(strat, cfg, window)
        rate = rate_for_threshold(probs, threshold)
        print(f"{f['fold_id']}: threshold={threshold}  rate={rate:.4%}  n_scored={len(probs)}")
        out[f["fold_id"]] = rate
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--fold", default=None, help="single fold_id; default every fold in the plan")
    ap.add_argument("--mode", choices=["pick", "control-rate"], default="pick")
    ap.add_argument("--target-rate", type=float, default=None,
                    help="required for --mode pick (default)")
    ap.add_argument("--threshold", type=float, default=None,
                    help="--mode control-rate only; default reads the plan's params_snapshot "
                         "buy.prediction_threshold")
    ap.add_argument("--out", default=None, help="write the fold_id -> value map here as JSON")
    a = ap.parse_args()

    plan = json.loads(Path(a.plan).read_text())

    if a.mode == "pick":
        if a.target_rate is None:
            print("ERROR: --mode pick requires --target-rate", file=sys.stderr)
            return 2
        out = cmd_pick(plan, a.fold, a.target_rate)
    else:
        out = cmd_control_rate(plan, a.fold, a.threshold)

    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=2) + "\n")
        print(f"wrote {a.out}")

    return 0 if out else 1


if __name__ == "__main__":
    raise SystemExit(main())
