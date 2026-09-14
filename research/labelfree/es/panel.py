"""Per-fold market panel and the recomputed entry/exit signals (T012, T020).

WHAT THIS BUILDS. One dict per fold holding, per pair and per bar: OHLCV, the indicator
columns the entry guards read, the stop's own rvol, the brief-1 cost, and the model's three
class probabilities from the side-car that es/signals.py regenerated.

WHY THE PROBABILITIES COME FROM A BACKTEST. The search recomputes entries at candidate guard
thresholds, so it needs p(buy) on EVERY bar -- not just the bars the incumbent entered on. The
alternative, running the 15 saved models offline, was weighed and rejected: it would be a
SECOND inference path, and this study already has a case where predict-time scaling silently
diverged from production (the single-task GAN scaler bug, 1.53pp, undetected until hunted).

NOTHING IS REPLAYED. `model_entry` and `model_exit` are both recomputed here from the three
probability columns, using the strategy's own rule:

    prediction = argmax(p) if max(p) > prediction_threshold else HOLD   (BaseNNStrategy:779)
    predict_buy  = prediction == BUY                                    (SELL 0, HOLD 1, BUY 2)
    predict_sell = prediction == SELL

An earlier design replayed `model_exit` from the archive and carried an 11.9% "invisible to
the surrogate" caveat. That caveat is GONE: the side-car makes the exit signal a function of
data we hold, so a parameter change that shifts when the model exits is now visible.

THE ENTRY CHAIN, mirroring BaseStrategy.populate_entry_trend (886-957) exactly:

    model_entry                                     p(buy) wins the argmax above threshold
  & volume > 0                                      MANDATORY, outside entry_enable_guards
  & rvol > entry_rvol_threshold                     MANDATORY, a tradeability constraint
  & atr_pct_roll > entry_atr_pct                    ┐
  & guard_metric < entry_guard_threshold            │ QUALITY guards, only when
  & close_norm  < entry_close_norm_threshold        │ entry_enable_guards is True
  & adx > entry_adx_threshold                       │
  & bb_width > entry_bb_width_threshold             ┘

Note `guard_metric <` — it is an UPPER bound, so RAISING entry_guard_threshold LOOSENS the
guard. Getting that backwards would make the search push the guard the wrong way.
"""
from __future__ import annotations

import hashlib
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.labelfree.costs import CostModel            # noqa: E402
from research.labelfree.folds import folds                # noqa: E402
from research.labelfree.es.signals import SINK_ROOT       # noqa: E402

CACHE_DIR = Path("/tmp/lf/es_panels")
SELL, HOLD, BUY = 0, 1, 2

# Guard thresholds NOT in the search space, pinned at the incumbent so the two that ARE
# searched are not confounded with the four that are not (Principle VIII, narrowed).
FIXED_GUARDS = {"entry_atr_pct": 0.006, "entry_close_norm_threshold": -0.1,
                "entry_adx_threshold": 38.0, "entry_bb_width_threshold": 0.06}
# The EXIT guard chain (BaseStrategy.populate_exit_trend, 1009-1015). Not searched, but it
# gates model_exit: without it the exit signal fires ~4x too often (measured f07 train: 143
# surrogate exits against 36 in a real backtest). Note the chain is gated on
# entry_enable_guards -- the ENTRY flag controls the EXIT guards, which BaseStrategy's own
# comment flags as deliberate-but-odd.
EXIT_GUARDS = {"exit_rvol_threshold": 2.0, "exit_guard_threshold": -0.1}
PREDICTION_THRESHOLD = 0.69          # NNNC_MLX.json buy.prediction_threshold

GUARD_COLS = ["rvol", "atr_pct_roll", "guard_metric", "close_norm", "adx", "bb_width"]


def _fold(fold_id: str):
    for f in folds():
        if f.fold_id == fold_id:
            return f
    raise KeyError(fold_id)


def load(fold_id: str, train: bool = True, sink_root: Path = SINK_ROOT,
         use_cache: bool = True) -> dict:
    """Build (or read from cache) one fold's panel.

    The cache key covers fold, side and the side-car's own content hash: a stale panel
    silently reused after the probabilities were regenerated would be undetectable, and
    feature 012's panel cache carries the same warning for the same reason.
    """
    fd = _fold(fold_id)
    which = "train" if train else "test"
    sink = Path(sink_root) / f"{fold_id}_{which}"
    if not sink.exists():
        raise SystemExit(
            f"no probability side-car at {sink}. Run:\n"
            f"  python -m research.labelfree.es.signals --folds {fold_id} --which {which}")

    sig = hashlib.sha1(
        ("|".join(sorted(f"{p.name}:{p.stat().st_size}:{int(p.stat().st_mtime)}"
                         for p in sorted(sink.glob('*.feather'))))).encode()).hexdigest()[:16]
    cache = CACHE_DIR / f"{fold_id}_{which}_{sig}.pkl"
    if use_cache and cache.exists():
        with cache.open("rb") as fh:
            return pickle.load(fh)

    from regime import regimedata as rd
    sys.path.insert(0, str(ROOT / "utils"))
    from DataframePopulator import DataframePopulator

    start, end = ((fd.train_start, fd.train_end) if train else (fd.test_start, fd.test_end))
    lo, hi = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")

    pairs: dict[str, dict] = {}
    for f in sorted(sink.glob("*.feather")):
        pair = f.stem.replace("_", "/")
        probs = pd.read_feather(f).set_index("date")

        df = rd.load_ohlcv(pair.split("/")[0])
        df = df[(df.index >= lo) & (df.index <= hi)]
        if df.empty:
            continue
        pop = DataframePopulator().add_minimal_indicators(df.reset_index().copy())
        pop.index = df.index[:len(pop)]

        # The stop's rvol is volume / rolling(20).mean() — NOT the `rvol` COLUMN, which is an
        # ewm(span=10). Same name, different quantity (BaseStrategy.py:1043-1048).
        vol = pop["volume"]
        rvol_stop = (vol / vol.rolling(20, min_periods=5).mean()).fillna(0.0)

        idx = pop.index.intersection(probs.index)
        if len(idx) < 200:
            continue
        pop, pr = pop.loc[idx], probs.loc[idx]
        cost = CostModel().per_unit(df.loc[idx], evaluation=True).to_numpy()

        cols = {c: pop[c].to_numpy(dtype=float) if c in pop else
                np.full(len(idx), np.nan) for c in GUARD_COLS}
        pairs[pair] = dict(
            index=idx,
            open=pop["open"].to_numpy(float), high=pop["high"].to_numpy(float),
            low=pop["low"].to_numpy(float), close=pop["close"].to_numpy(float),
            volume=vol.loc[idx].to_numpy(float),
            rvol_stop=rvol_stop.loc[idx].to_numpy(float),
            cost=cost,
            p_sell=pr["%predict_prob_sell"].to_numpy(float),
            p_hold=pr["%predict_prob_hold"].to_numpy(float),
            p_buy=pr["%predict_prob_buy"].to_numpy(float),
            **cols)

    if not pairs:
        raise SystemExit(f"{fold_id}/{which}: no usable pairs")

    out = {"fold_id": fold_id, "which": which, "pairs": pairs,
           "span_years": _span_years(pairs)}
    out = apply_leakage_control(out, fd)
    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with cache.open("wb") as fh:
            pickle.dump(out, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return out


def _span_years(pairs: dict) -> float:
    n = max(len(p["open"]) for p in pairs.values())
    return n / (365.25 * 96)


def apply_leakage_control(panel: dict, fd) -> dict:
    """Drop the fold's purge window from the END of a training panel (T020).

    ASSERTED, not assumed. This study had `purge_bars = 0` configured everywhere with
    `apply_leakage_control` never actually called, and the absence was invisible because the
    configuration looked right. The count applied is recorded ON the panel so a reader can
    see what happened rather than trust that something did.

    A trade opened inside the purge window would run past train_end into the test range, which
    is the concrete leak this prevents. `max_hold` bounds a trade at cexit_max_days, and the
    search may raise that to 28 days, so the purge must cover the LARGEST reachable hold.
    """
    from research.labelfree.es.space import DIMS
    max_days = next(d.hi for d in DIMS if d.name == "cexit_max_days")
    purge = int(max_days * 96) if panel["which"] == "train" else 0
    purge = max(purge, int(getattr(fd, "purge_bars", 0) or 0))

    MIN_BARS = 96 * 30          # a month; below this a fold's pair cannot carry a trade set
    dropped = []
    if purge:
        for pair in list(panel["pairs"]):
            d = panel["pairs"][pair]
            keep = max(0, len(d["open"]) - purge)
            if keep < MIN_BARS:
                # A pair the purge empties must be REMOVED, not left at length 0. A
                # zero-length pair yields a NaN exposure, and `nan < floor` is False, so it
                # would pass the participation floor silently.
                dropped.append(pair)
                del panel["pairs"][pair]
                continue
            for k, v in list(d.items()):
                d[k] = v[:keep]
        panel["span_years"] = _span_years(panel["pairs"])
    panel["pairs_dropped_by_purge"] = dropped
    panel["purge_bars_applied"] = purge
    panel["leakage_control_called"] = True
    return panel


# ------------------------------------------------------------------- recomputed signals

def model_signals(d: dict, threshold: float = PREDICTION_THRESHOLD):
    """(buy, sell) boolean arrays, from BaseNNStrategy.argmax_with_threshold (779-783)."""
    P = np.stack([d["p_sell"], d["p_hold"], d["p_buy"]], axis=1)
    cls = np.where(P.max(axis=1) > threshold, P.argmax(axis=1), HOLD)
    return cls == BUY, cls == SELL


def exits(d: dict, enable_guards: bool = True) -> np.ndarray:
    """The exit chain of BaseStrategy.populate_exit_trend (995-1017), recomputed.

    `model_exit` is NOT just `predict_sell`: it is gated by the exit rvol and guard_metric
    thresholds. Omitting them made the surrogate exit on the model far too often.
    """
    _buy, sell = model_signals(d)
    cond = sell
    if enable_guards:
        cond &= d["rvol"] > EXIT_GUARDS["exit_rvol_threshold"]
        # LOWER bound here, unlike the entry guard's upper bound on the same column.
        cond &= d["guard_metric"] > EXIT_GUARDS["exit_guard_threshold"]
    return np.nan_to_num(cond.astype(float), nan=0.0).astype(bool)


def entries(d: dict, params: dict, enable_guards: bool = True) -> np.ndarray:
    """The entry chain of BaseStrategy.populate_entry_trend (886-957), recomputed."""
    buy, _ = model_signals(d)
    cond = buy & (d["volume"] > 0.0) & (d["rvol"] > params["entry_rvol_threshold"])
    if enable_guards:
        cond &= d["atr_pct_roll"] > FIXED_GUARDS["entry_atr_pct"]
        # UPPER bound: raising the threshold LOOSENS this guard.
        cond &= d["guard_metric"] < params["entry_guard_threshold"]
        cond &= d["close_norm"] < FIXED_GUARDS["entry_close_norm_threshold"]
        cond &= d["adx"] > FIXED_GUARDS["entry_adx_threshold"]
        cond &= d["bb_width"] > FIXED_GUARDS["entry_bb_width_threshold"]
    return np.nan_to_num(cond.astype(float), nan=0.0).astype(bool)
