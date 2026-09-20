# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0303, C0411, C0413
# pylint: disable=import-error
# flake8: noqa: F401, E402

"""
`research/clfstudy/run.py` -- run one (config, window, seed) experiment (T017).

`run_one()` is the single place that wires data.py -> splits.py -> features.py ->
windows.py -> registry.py -> metrics.py into one `RunResult` row (data-model.md
§RunResult) and appends it to `results/results.jsonl`. It is called directly by
`phase.py` (one call per expanded run key) and by `calibrate.py` (timing), and via the
CLI below for one-off / `--dry-run` invocations.

Leakage boundary: `scaler`/`class_weights` are fit only on the concatenated `train`-role
rows of the *included* pairs for this window (features.fit_scaler/class_weights);
role-selection and windowing are otherwise entirely owned by splits.py/windows.py.

Opt-in `--save-probs` writes the test-row predictions (one row per `X_test`/`proba` row) to
`PROBS_DIR/<config_id>_<window_id>_<seed>.parquet` via pyarrow, after scoring. `probs/` is
scratch/debug output, not part of the tracked study artifacts -- it should be gitignored.
"""
from __future__ import annotations

import argparse
import fcntl
import gc
import json
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlx.core as mx
import numpy as np
import pandas as pd
import yaml

from research.clfstudy import (
    CACHE_DIR,
    HARNESS_VERSION,
    PKG,
    RESULTS_PATH,
    ROOT,
    SPLITS_DIR,
    assert_interpreter,
)
from research.clfstudy import baselines, data, features, labels, metrics, registry, splits, windows
from research.clfstudy.registry import ExperimentConfig

# ---------------------------------------------------------------------------
# Process-level caches
# ---------------------------------------------------------------------------

#: `Optional[Dict[str, pd.DataFrame]]` (pair -> frame), exactly as before `label` existed --
#: kept a flat dict, not nested by label, so `smoke.py`'s `run_mod._FRAMES_CACHE = data
#: .load_all(pairs=...)` (a direct assignment, bypassing `_load_frames`) and the study's
#: tests' `monkeypatch.setattr(run_mod, "_FRAMES_CACHE", None)` both keep working
#: unchanged. `_FRAMES_CACHE_LABEL_KEY` records which label the cache holds -- defaults to
#: `"gbb"` (`labels.label_key(None)`) so a directly-assigned `_FRAMES_CACHE` (always built
#: for the default label today) is trusted until a DIFFERENT label is actually requested.
_FRAMES_CACHE: Optional[Dict[str, pd.DataFrame]] = None
_FRAMES_CACHE_LABEL_KEY: str = "gbb"
_SPLITSET_CACHE: Dict[str, splits.SplitSet] = {}

#: `--save-probs` output directory (scratch/debug, gitignore this); a fresh dir per repo
#: checkout since nothing under it is a tracked study artifact.
PROBS_DIR = PKG / "probs"

_METRIC_KEYS = (
    "bal_acc", "macro_f1", "mcc",
    "prec_sell", "prec_hold", "prec_buy", "rec_sell", "rec_hold", "rec_buy",
    "logloss", "auroc_sell", "auroc_hold", "auroc_buy", "auroc_macro", "ece",
    "reliability_bins", "confusion", "buy_prec_at_08", "buy_cov_at_08",
    "trig_buy_n", "trig_buy_base_rate", "trig_buy_prec_argmax", "trig_buy_auroc",
    "trig_buy_mcc",
    "trig_sell_n", "trig_sell_base_rate", "trig_sell_prec_argmax", "trig_sell_auroc",
    "trig_sell_mcc",
    "trig_rule_mcc",
)

_METRIC_DEFAULTS: Dict[str, Any] = {
    "bal_acc": float("nan"), "macro_f1": float("nan"), "mcc": float("nan"),
    "prec_sell": float("nan"), "prec_hold": float("nan"), "prec_buy": float("nan"),
    "rec_sell": float("nan"), "rec_hold": float("nan"), "rec_buy": float("nan"),
    "logloss": float("nan"),
    "auroc_sell": float("nan"), "auroc_hold": float("nan"), "auroc_buy": float("nan"),
    "auroc_macro": float("nan"), "ece": float("nan"), "reliability_bins": [],
    "confusion": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
    "buy_prec_at_08": float("nan"), "buy_cov_at_08": float("nan"),
    "trig_buy_n": float("nan"), "trig_buy_base_rate": float("nan"),
    "trig_buy_prec_argmax": float("nan"), "trig_buy_auroc": float("nan"),
    "trig_buy_mcc": float("nan"),
    "trig_sell_n": float("nan"), "trig_sell_base_rate": float("nan"),
    "trig_sell_prec_argmax": float("nan"), "trig_sell_auroc": float("nan"),
    "trig_sell_mcc": float("nan"),
    "trig_rule_mcc": float("nan"),
}


def _load_frames(label: Optional[Dict[str, Any]] = None) -> Dict[str, pd.DataFrame]:
    """`data.load_all(label=label)`, cached at module level for the life of this
    process. Reloads whenever the cache is unset (`None`) OR was built for a different
    label (`_FRAMES_CACHE_LABEL_KEY`) -- so switching `cfg.label` between runs in the
    same process (e.g. a phase mixing gbb and triple_barrier configs) never serves one
    label's frames to the other."""
    global _FRAMES_CACHE, _FRAMES_CACHE_LABEL_KEY
    lkey = labels.label_key(label)
    if _FRAMES_CACHE is None or _FRAMES_CACHE_LABEL_KEY != lkey:
        _FRAMES_CACHE = data.load_all(label=label)
        _FRAMES_CACHE_LABEL_KEY = lkey
    return _FRAMES_CACHE


def load_or_build_splitset(split_set: str) -> splits.SplitSet:
    """`splits.load(SPLITS_DIR/<split_set>.json)`, building+saving it first if absent.

    Cached at module level (shared with `phase.py`/`calibrate.py` via this function).
    """
    if split_set in _SPLITSET_CACHE:
        return _SPLITSET_CACHE[split_set]
    path = SPLITS_DIR / f"{split_set}.json"
    if path.exists():
        ss = splits.load(path)
    else:
        ss = splits.build(split_set)
        splits.validate(ss)
        splits.save(ss, path)
    _SPLITSET_CACHE[split_set] = ss
    return ss


def _git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        if result.returncode == 0:
            sha = result.stdout.strip()
            if sha:
                return sha
    except Exception:
        pass
    return "unknown"


def _mx_get(name: str):
    """`mx.<name>` if this MLX build exposes it at the top level, else `mx.metal.<name>`
    if that (older-API) location has it, else `None`."""
    fn = getattr(mx, name, None)
    if fn is not None:
        return fn
    metal = getattr(mx, "metal", None)
    return getattr(metal, name, None) if metal is not None else None


def _release_mlx_memory() -> float:
    """`gc.collect()` (so MLX arrays with no remaining Python references are freed) then
    release MLX's Metal buffer cache back to the OS, log a one-line memory probe, and reset
    the peak-memory counter for the next run.

    Long-lived processes (`phase.py --no-isolate`) run 100+ of these back to back; without
    this, freed-but-cached Metal buffers accumulate across runs (they don't show up in RSS)
    until the OS is forced to swap. Returns peak active memory in MB, or NaN if this MLX
    build exposes none of the memory-introspection APIs.
    """
    gc.collect()

    clear_cache = _mx_get("clear_cache")
    if clear_cache is not None:
        clear_cache()

    def _mb(fn) -> Optional[float]:
        if fn is None:
            return None
        return fn() / (1024 * 1024)

    active_mb = _mb(_mx_get("get_active_memory"))
    peak_mb = _mb(_mx_get("get_peak_memory"))
    cache_mb = _mb(_mx_get("get_cache_memory"))

    def _fmt(v: Optional[float]) -> str:
        return "n/a" if v is None else f"{v:.1f}MB"

    print(f"mem: active={_fmt(active_mb)} peak={_fmt(peak_mb)} cache={_fmt(cache_mb)}")

    reset_peak = _mx_get("reset_peak_memory")
    if reset_peak is not None:
        reset_peak()

    return float(peak_mb) if peak_mb is not None else float("nan")


def _sanitize(value: Any) -> Any:
    """Recursively replace NaN floats with `None`; leave everything else as-is."""
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, dict):
        return {k: _sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(v) for v in value]
    return value


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"object of type {type(obj)!r} is not JSON serializable")


def _append_result(row: Dict[str, Any]) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(_sanitize(row), default=_json_default)
    with open(RESULTS_PATH, "a", encoding="utf-8") as f:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        except OSError:
            pass
        try:
            f.write(line + "\n")
            f.flush()
        finally:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass


def _print_summary(row: Dict[str, Any]) -> None:
    print(
        f"window={row['window_id']} seed={row['seed']} status={row['status']} "
        f"mcc={row['mcc']:.4f} macro_f1={row['macro_f1']:.4f} bal_acc={row['bal_acc']:.4f} "
        f"epochs_run={row['epochs_run']} train_seconds={row['train_seconds']:.1f} "
        f"n_params={row['n_params']} ckpt={row['ckpt_name']}"
    )


def _build_tensors(cfg: ExperimentConfig, window_id: str) -> Dict[str, Any]:
    """Window lookup -> included/excluded pairs -> per-window scaler/class-weights ->
    train/val/test tensors (research R4). Raises on any structural problem (unknown
    window, no training rows, ...); the caller decides whether that becomes a `failed`
    row (real runs) or an uncaught traceback (`--dry-run`, a debugging tool)."""
    frames = _load_frames(cfg.label)
    ss = load_or_build_splitset(cfg.split_set)
    w = next((win for win in ss.windows if win.window_id == window_id), None)
    if w is None:
        raise ValueError(f"window {window_id!r} not found in split set {cfg.split_set!r}")

    first_dates = {pair: pd.Timestamp(frame["date"].min()) for pair, frame in frames.items()}
    included = splits.included_pairs(w, first_dates)
    excluded = [p for p in frames if p not in included]

    cols = features.columns_for(cfg.input_set)

    roles = {pair: splits.assign_roles(frames[pair]["date"], w) for pair in included}

    train_parts = [
        frames[pair].loc[roles[pair] == "train", cols + ["label"]] for pair in included
    ]
    train_df = (
        pd.concat(train_parts, axis=0)
        if train_parts
        else pd.DataFrame(columns=cols + ["label"])
    )
    if len(train_df) == 0:
        raise ValueError(
            f"window {window_id!r}: no training rows for any included pair "
            f"(included={included})"
        )

    scaler = features.fit_scaler(train_df, cols, mode=cfg.scaler_mode)
    diag = getattr(
        scaler, "clfstudy_diag",
        {"degenerate_iqr_cols": [], "constant_cols": [], "mode": cfg.scaler_mode},
    )
    cw = features.class_weights(train_df["label"].to_numpy())

    # Zero-volume contamination (research: FIX for "contamination must be visible"):
    # freqtrade fills a missing candle with volume=0, OHLC=prev close. Read the RAW
    # `volume` column (present on every `data.load_all()` frame) BEFORE any
    # scaling/column selection would drop it; a caller-supplied frame without it (e.g. a
    # synthetic test frame) falls back to NaN, same pattern as guard_metric/bb_width above
    # -- `zero_volume_share` excludes NaN rows from its ratio, reporting NaN (not a
    # misleading 0.0) when no real volume data is present at all.
    train_vol_parts, test_vol_parts = [], []
    for pair in included:
        frame = frames[pair]
        role = roles[pair]
        vol = (
            frame["volume"].to_numpy(dtype=np.float64) if "volume" in frame.columns
            else np.full(len(frame), np.nan)
        )
        train_vol_parts.append(vol[role == "train"])
        test_vol_parts.append(vol[role == "test"])
    train_vol = np.concatenate(train_vol_parts) if train_vol_parts else np.empty(0)
    test_vol = np.concatenate(test_vol_parts) if test_vol_parts else np.empty(0)
    zero_vol_share_train = data.zero_volume_share(pd.DataFrame({"volume": train_vol}))
    zero_vol_share_test = data.zero_volume_share(pd.DataFrame({"volume": test_vol}))

    scaled_frames: Dict[str, pd.DataFrame] = {}
    for pair in included:
        frame = frames[pair]
        scaled = features.apply_scaler(frame, scaler, cols)
        scaled["date"] = frame["date"].to_numpy()
        scaled["label"] = frame["label"].to_numpy()
        # RAW (pre-scaling) guard_metric/bb_width, under names distinct from any (possibly
        # scaled) same-named column already in `scaled` (e.g. `bb_width` is itself in IND24
        # and gets RobustScaler-transformed above) -- only used downstream to reconstruct the
        # gbb label's deterministic trigger rule (metrics.trigger_masks/score), never fed to
        # a model. `data.load_all()` frames always carry both (DataframePopulator's MINIMAL
        # indicator set); a caller-supplied frame that doesn't (e.g. a synthetic test frame
        # built without them) falls back to NaN, which trivially fails every trigger
        # comparison -- trig_* metrics become "no trigger rows", never a crash.
        n_rows = len(frame)
        scaled["_guard_metric_raw"] = (
            frame["guard_metric"].to_numpy() if "guard_metric" in frame.columns
            else np.full(n_rows, np.nan)
        )
        scaled["_bb_width_raw"] = (
            frame["bb_width"].to_numpy() if "bb_width" in frame.columns
            else np.full(n_rows, np.nan)
        )
        scaled_frames[pair] = scaled

    tensors = windows.make_tensors(
        scaled_frames, roles, cols, cfg.seq_len, stride=cfg.train["stride"], label_col="label",
        guard_metric_col="_guard_metric_raw", bb_width_col="_bb_width_raw",
    )
    del scaled_frames

    X_train, y_train_1h, y_train_int = tensors["train"]
    X_val, y_val_1h, y_val_int = tensors["val"]
    X_test, y_test_int, test_dates, test_pairs = tensors["test"]
    test_guard_metric, test_bb_width = tensors["test_trigger"]
    del tensors

    return {
        "included": included,
        "excluded": excluded,
        "cols": cols,
        "cw": cw,
        "degenerate_iqr_cols": diag["degenerate_iqr_cols"],
        "constant_cols": diag["constant_cols"],
        "zero_vol_share_train": zero_vol_share_train,
        "zero_vol_share_test": zero_vol_share_test,
        "X_train": X_train, "y_train_1h": y_train_1h, "y_train_int": y_train_int,
        "X_val": X_val, "y_val_1h": y_val_1h, "y_val_int": y_val_int,
        "X_test": X_test, "y_test_int": y_test_int,
        "test_dates": test_dates, "test_pairs": test_pairs,
        "test_guard_metric": test_guard_metric, "test_bb_width": test_bb_width,
    }


def _clear_stale_model_files(clf) -> int:
    """Remove any pre-existing weight/checkpoint files at `clf.model_path` and
    `clf.get_checkpoint_path()` (plus sibling `<stem>.*` files in the same directory,
    e.g. metadata written alongside the weights by `save()`), so `train()` always starts
    from a freshly-initialised model.

    `MLXClassifierNary.train()` calls `self.model = self.load()` whenever `self.model` is
    `None`, regardless of `force_train` -- so a leftover file from an earlier run of the
    same `(config_id, seed)` (a retry, a resume after the phase runner was killed, or a
    config whose id didn't change although its architecture did) silently warm-starts
    training, or fails to load if the architecture changed. Every study run must train
    from scratch, so this is called right after `registry.build()` and before
    `clf.train()`. Returns the number of files removed.
    """
    stale_paths = {Path(clf.model_path), Path(clf.get_checkpoint_path())}
    for stale in list(stale_paths):
        stale_paths.update(stale.parent.glob(f"{stale.stem}.*"))

    removed = 0
    for stale in sorted(stale_paths):
        if stale.exists():
            stale.unlink()
            removed += 1
            print(f"fresh-start: removed stale {stale}")
    return removed


def _write_probs(
    cid: str,
    window_id: str,
    seed: int,
    test_dates: np.ndarray,
    test_pairs: np.ndarray,
    y_test_int: np.ndarray,
    proba: np.ndarray,
    test_guard_metric: np.ndarray,
    test_bb_width: np.ndarray,
    buy_mask: np.ndarray,
    sell_mask: np.ndarray,
) -> str:
    """Write one row per test example (same row order as `X_test`/`proba`, both built from
    the same per-pair concatenation in `windows.make_tensors`) to
    `PROBS_DIR/<cid>_<window_id>_<seed>.parquet` via pyarrow, atomically (write to a `.tmp`
    sibling then `rename`, so a killed-mid-write process never leaves a corrupt parquet at
    the real path). Returns the destination path relative to `PKG`.
    """
    PROBS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "date": pd.to_datetime(np.asarray(test_dates), utc=True),
        "pair": np.asarray(test_pairs, dtype=object),
        "label": np.asarray(y_test_int).astype(int),
        "p_sell": proba[:, 0].astype(np.float32),
        "p_hold": proba[:, 1].astype(np.float32),
        "p_buy": proba[:, 2].astype(np.float32),
        "guard_metric": np.asarray(test_guard_metric, dtype=float),
        "bb_width": np.asarray(test_bb_width, dtype=float),
        "buy_trigger": np.asarray(buy_mask, dtype=bool),
        "sell_trigger": np.asarray(sell_mask, dtype=bool),
    })
    dest = PROBS_DIR / f"{cid}_{window_id}_{seed}.parquet"
    tmp = dest.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False, engine="pyarrow")
    tmp.replace(dest)
    return str(dest.relative_to(PKG))


def run_one(
    cfg: ExperimentConfig,
    window_id: str,
    seed: int,
    save_weights: bool = False,
    save_probs: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Run one (config, window, seed) experiment; append one `RunResult` row unless
    `dry_run`. Returns the row dict (empty dict for a dry run).

    `save_probs`: after scoring, additionally write the test-row predictions to
    `PROBS_DIR/<config_id>_<window_id>_<seed>.parquet` (see `_write_probs`); the row's
    `probs_path` field records the (relative) path, or stays `None` if not saved."""
    started_at = datetime.now(timezone.utc)
    cid = registry.config_id(cfg)

    if dry_run:
        prep = _build_tensors(cfg, window_id)
        n_train = len(prep["X_train"])
        n_val = len(prep["X_val"])
        n_test = len(prep["X_test"])
        class_counts_train = np.bincount(prep["y_train_int"], minlength=3)[:3].tolist()
        print(f"[dry-run] config_id={cid} window={window_id} seed={seed}")
        print(f"  pairs_included={prep['included']}")
        print(f"  pairs_excluded={prep['excluded']}")
        print(f"  n_train={n_train} n_val={n_val} n_test={n_test}")
        print(
            f"  X_train.shape={prep['X_train'].shape} X_val.shape={prep['X_val'].shape} "
            f"X_test.shape={prep['X_test'].shape}"
        )
        print(f"  class_counts_train={class_counts_train}")
        return {}

    status = "ok"
    error: Optional[str] = None
    included: List[str] = []
    excluded: List[str] = []
    n_train = n_val = n_test = 0
    class_counts_train = [0, 0, 0]
    class_counts_test = [0, 0, 0]
    pred_counts = [0, 0, 0]
    n_params = 0
    train_seconds = 0.0
    epochs_run = 0
    best_epoch = 0
    weights_path: Optional[str] = None
    probs_path: Optional[str] = None
    ckpt_name: Optional[str] = None
    stale_files_removed = 0
    degenerate_iqr_cols: List[str] = []
    constant_cols: List[str] = []
    zero_vol_share_train = float("nan")
    zero_vol_share_test = float("nan")
    timing_stats = {
        "infer_ms_per_window_b256": float("nan"),
        "infer_ms_per_window_b1": float("nan"),
    }
    metric_fields = dict(_METRIC_DEFAULTS)
    mlx_peak_mb = float("nan")

    try:
        prep = _build_tensors(cfg, window_id)
        included, excluded, cols, cw = prep["included"], prep["excluded"], prep["cols"], prep["cw"]
        degenerate_iqr_cols = prep["degenerate_iqr_cols"]
        constant_cols = prep["constant_cols"]
        zero_vol_share_train = prep["zero_vol_share_train"]
        zero_vol_share_test = prep["zero_vol_share_test"]
        X_train, y_train_1h, y_train_int = prep["X_train"], prep["y_train_1h"], prep["y_train_int"]
        X_val, y_val_1h = prep["X_val"], prep["y_val_1h"]
        X_test, y_test_int = prep["X_test"], prep["y_test_int"]
        test_dates, test_pairs = prep["test_dates"], prep["test_pairs"]
        test_guard_metric, test_bb_width = prep["test_guard_metric"], prep["test_bb_width"]

        n_train, n_val, n_test = len(X_train), len(X_val), len(X_test)
        class_counts_train = np.bincount(y_train_int, minlength=3)[:3].tolist()
        class_counts_test = np.bincount(y_test_int, minlength=3)[:3].tolist()

        if registry.is_baseline(cfg.arch):
            if cfg.arch == "MAJORITY":
                proba, _note = baselines.majority(y_train_int, n_test)
            elif cfg.arch == "PERSISTENCE":
                proba, _note = baselines.persistence(y_test_int)
            else:
                raise ValueError(f"unknown baseline arch {cfg.arch!r}")
        else:
            clf = registry.build(
                cfg, nf=len(cols), seq_len=cfg.seq_len, seed=seed,
                checkpoint_dir=CACHE_DIR / "ckpt", window_id=window_id,
            )
            ckpt_name = Path(clf.model_path).name
            stale_files_removed = _clear_stale_model_files(clf)
            t0 = time.perf_counter()
            clf.train(
                X_train, X_val, y_train_1h, y_val_1h,
                force_train=True, class_weights=cw,
            )
            train_seconds = time.perf_counter() - t0
            n_params = registry.param_count(clf)
            epochs_run = clf.epochs_run
            best_epoch = clf.best_epoch

            if epochs_run < 1:
                raise RuntimeError("no epochs run")

            proba = clf.predict(X_test)
            timing_stats = metrics.timing(clf.predict, X_test)

            if save_weights:
                dest_dir = PKG / "models"
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest = dest_dir / f"{cid}_{window_id}_{seed}.safetensors"
                shutil.copy(clf.model_path, dest)
                weights_path = str(dest)

        if not np.isfinite(proba).all():
            raise ValueError("non-finite values in predicted probabilities")

        buy_mask, sell_mask = metrics.trigger_masks(test_guard_metric, test_bb_width)
        m = metrics.score(y_test_int, proba, buy_mask=buy_mask, sell_mask=sell_mask)
        metric_fields = {k: m[k] for k in _METRIC_KEYS}
        pred_counts = m["pred_counts"]
        class_counts_test = m["class_counts"]
        status = "collapsed" if m["collapsed"] else ("class_absent" if m["class_absent"] else "ok")

        if save_probs:
            probs_path = _write_probs(
                cid, window_id, seed, test_dates, test_pairs, y_test_int, proba,
                test_guard_metric, test_bb_width, buy_mask, sell_mask,
            )
    except Exception:
        status = "failed"
        error = traceback.format_exc()
    finally:
        # Drop every reference to the (potentially multi-GB) tensors, the trained
        # model and its predictions before assembling/serialising the row, then hand
        # MLX's Metal buffer cache back to the OS. A NameError here just means the
        # corresponding name never got bound (e.g. `_build_tensors` failed before
        # `prep`, or a baseline arch never bound `clf`).
        try:
            del prep, X_train, y_train_1h, y_train_int, X_val, y_val_1h, X_test, y_test_int
            del test_dates, test_pairs
        except NameError:
            pass
        try:
            del clf
        except NameError:
            pass
        try:
            del proba
        except NameError:
            pass
        mlx_peak_mb = _release_mlx_memory()

    finished_at = datetime.now(timezone.utc)

    row: Dict[str, Any] = {
        "config_id": cid,
        "split_set": cfg.split_set,
        "window_id": window_id,
        "seed": seed,
        "phase": cfg.phase,
        "arch": cfg.arch,
        "hp": cfg.hp,
        "input_set": cfg.input_set,
        "seq_len": cfg.seq_len,
        "train": cfg.train,
        "label": cfg.label,
        "tags": cfg.tags,
        "scaler_mode": cfg.scaler_mode,
        "status": status,
        "error": error,
        "pairs_included": included,
        "pairs_excluded": excluded,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "class_counts_train": class_counts_train,
        "class_counts_test": class_counts_test,
        "pred_counts": pred_counts,
        "degenerate_iqr_cols": degenerate_iqr_cols,
        "constant_cols": constant_cols,
        "zero_vol_share_train": zero_vol_share_train,
        "zero_vol_share_test": zero_vol_share_test,
        **{k: metric_fields[k] for k in _METRIC_KEYS},
        "n_params": n_params,
        "train_seconds": train_seconds,
        "epochs_run": epochs_run,
        "best_epoch": best_epoch,
        "infer_ms_per_window_b256": timing_stats["infer_ms_per_window_b256"],
        "infer_ms_per_window_b1": timing_stats["infer_ms_per_window_b1"],
        "mlx_peak_mb": mlx_peak_mb,
        "stale_files_removed": stale_files_removed,
        "harness_version": HARNESS_VERSION,
        "git_sha": _git_sha(),
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "weights_path": weights_path,
        "probs_path": probs_path,
        "ckpt_name": ckpt_name,
    }

    _append_result(row)
    _print_summary(row)
    return row


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_config(arg: str) -> ExperimentConfig:
    text = arg.strip()
    if text.startswith("{"):
        payload = json.loads(text)
    else:
        with open(text, "r", encoding="utf-8") as f:
            payload = yaml.safe_load(f)
    cfg = ExperimentConfig(
        arch=payload["arch"],
        hp=payload.get("hp", {}),
        input_set=payload.get("input_set", "ind24"),
        seq_len=payload.get("seq_len", 16),
        split_set=payload.get("split_set", "v1"),
        train=payload.get("train", {}),
        phase=payload.get("phase", 0),
        seeds=payload.get("seeds", [0]),
        tags=payload.get("tags", []),
        label=payload.get("label", dict(registry.DEFAULT_LABEL)),
        scaler_mode=payload.get("scaler_mode", registry.DEFAULT_SCALER_MODE),
    )
    cfg.validate()
    return cfg


def main(argv: Optional[List[str]] = None) -> int:
    try:
        assert_interpreter()
    except SystemExit as exc:
        print(str(exc), file=sys.stderr)
        return 3

    parser = argparse.ArgumentParser(prog="python -m research.clfstudy.run")
    parser.add_argument("--config", required=True, help="YAML file path or inline JSON")
    parser.add_argument("--window", required=True, choices=["A", "B", "C", "all"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--save-weights", action="store_true")
    parser.add_argument("--save-probs", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    try:
        cfg = _load_config(args.config)
    except Exception as exc:
        print(f"invalid config: {exc}", file=sys.stderr)
        return 2

    try:
        ss = load_or_build_splitset(cfg.split_set)
    except Exception as exc:
        print(f"invalid split set {cfg.split_set!r}: {exc}", file=sys.stderr)
        return 2

    window_ids = [w.window_id for w in ss.windows] if args.window == "all" else [args.window]

    any_failed = False
    for window_id in window_ids:
        row = run_one(
            cfg, window_id, args.seed,
            save_weights=args.save_weights, save_probs=args.save_probs, dry_run=args.dry_run,
        )
        if row.get("status") == "failed":
            any_failed = True

    return 1 if any_failed else 0


if __name__ == "__main__":
    sys.exit(main())
