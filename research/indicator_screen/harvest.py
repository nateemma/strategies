"""037 harvest (research R1, R5, R13): one causal, cached frame per pair holding every SimpleStrategies
indicator column and raw signal, the ARO control's 24 model inputs and the gbb candidate flag.

- base frame: the SimpleStrategy populate path (precision columns, MINIMAL indicators,
  SimpleStrategy.add_additional_indicators, debug indicators) -> columns `base:<col>`;
- per in-scope class: a copy of the base with atr_ok forced True, the class's own signal code via
  overrides.raw_signals -> new or changed columns `<class>:<col>`, raw signals
  `raw_buy:<class>[:variant]` / `raw_sell:...` (int8);
- `aro:<col>`: the ARO control's include_list from its own indicator step (FR-006); a column equal
  to a base column is recorded as an alias, not stored twice;
- `base:gbb_candidate`: the registered gbb trigger on the frame its own populate step builds (R5).

Constant or all-missing columns are excluded with a reason; a class that raises is `signal_error`.
Columns that FAIL the truncation test (causality.json) are dropped, reason `lookahead`.

    PYTHONPATH=. ../../.venv/bin/python -m research.indicator_screen.harvest --time-one SOL/USDT
    PYTHONPATH=. ../../.venv/bin/python -m research.indicator_screen.harvest --pair all
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

STRAT = Path(__file__).resolve().parents[2]
ROOT = STRAT.parents[1]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(STRAT), str(STRAT / "evaluation")]

from research.indicator_screen import discover as D  # noqa: E402
from research.indicator_screen import overrides as O  # noqa: E402
from research.indicator_screen import windows as W  # noqa: E402

CONFIG = "user_data/strategies/config/config_wf.json"
CACHE = ROOT / "user_data" / "data" / "indicator_screen"
MANIFEST = HERE / "manifest.json"
CAUSALITY = HERE / "causality.json"
ARO_CLASS = "NNNC_A_Raw_Open"
BTC = "BTC/USDT"
OHLCV = ["open", "high", "low", "close", "volume"]
#: base-frame columns that are filters/bookkeeping, not indicators
BASE_SKIP = {"date", *OHLCV, "atr_ok", "buy_signal", "sell_signal", "entry_signals",
             "exit_signals", "enter_long", "exit_long", "enter_tag", "exit_tag",
             # check_precision_columns bookkeeping (decimal-place counts), not indicators
             "open_count", "high_count", "low_count", "close_count", "max_count"}

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------- loading


def _quiet():
    warnings.filterwarnings("ignore")
    logging.getLogger("freqtrade").setLevel(logging.WARNING)


def base_config():
    from freqtrade.configuration import Configuration
    from freqtrade.enums import RunMode

    args = {"config": [str(ROOT / CONFIG)], "strategy": "CCI",
            "user_data_dir": ROOT / "user_data", "strategy_path": str(STRAT / "SimpleStrategies")}
    return Configuration(args, RunMode.BACKTEST).get_config()


def load_strategy(cfg, name: str):
    """The class with its current parameters, through freqtrade's own resolver (R1)."""
    from freqtrade.resolvers import StrategyResolver

    c = copy.deepcopy(cfg)
    c["strategy"] = name
    s = StrategyResolver.load_strategy(c)
    s.dbg_verbose = False
    return s


def load_aro():
    import pick_threshold as P

    strat, _ = P._load_fold_context(ARO_CLASS, "NNNC", CONFIG)
    return strat


def pairs(cfg=None) -> list[str]:
    cfg = cfg or base_config()
    return list(cfg["exchange"]["pair_whitelist"]) + [BTC]


#: a bar is scored only if it exists in the raw data AND >= this share of the trailing WARMUP_BARS
#: do (XRP's 2021-01 -> 2023-07 gap is gap-filled flat candles, not data: spec edge case)
DATA_OK_SHARE = 0.95


def load_ohlcv(cfg, pair: str, filled: bool = True) -> pd.DataFrame:
    """15m candles [DATA_START, DATA_END), exactly as freqtrade's loader returns them (gap-filled
    unless `filled` is False)."""
    from freqtrade.configuration import TimeRange
    from freqtrade.data import history

    tr = TimeRange.parse_timerange(f"{W.DATA_START.replace('-', '')}-{W.DATA_END.replace('-', '')}")
    df = history.load_pair_history(pair=pair, timeframe="15m", datadir=cfg["datadir"],
                                   timerange=tr, fill_up_missing=filled,
                                   data_format=cfg.get("dataformat_ohlcv", "feather"),
                                   candle_type=cfg.get("candle_type_def", "spot"))
    return df[df["date"] < pd.Timestamp(W.DATA_END, tz="UTC")].reset_index(drop=True)


def data_flags(dates: pd.Series, raw_dates: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """(has_data, data_ok) int8: the bar is in the raw file; and the trailing WARMUP_BARS are
    >= DATA_OK_SHARE real (so indicators are warm on real bars, not on filled ones)."""
    has = pd.Series(pd.Series(dates).isin(pd.Series(raw_dates)).to_numpy(float))
    share = has.rolling(W.WARMUP_BARS, min_periods=W.WARMUP_BARS).mean()
    ok = (has > 0) & (share >= DATA_OK_SHARE)
    return has.to_numpy(np.int8), ok.to_numpy(np.int8)


# ---------------------------------------------------------------------------- frame building


def populate_path(strat, ohlcv: pd.DataFrame, pair: str) -> pd.DataFrame:
    """BaseStrategy.populate_indicators' steps, without any training/prediction hook."""
    from utils.DataframePopulator import DatasetType

    strat.curr_pair = pair
    strat.iteration_init()
    d = strat.check_precision_columns(ohlcv.copy())
    d = strat.dataframePopulator.add_indicators(d, dataset_type=DatasetType.MINIMAL)
    d = strat.add_additional_indicators(d)
    return strat.add_debug_indicators(d)


def gbb_candidate(ohlcv: pd.DataFrame) -> np.ndarray:
    """The registered gbb trigger on the frame its own populate step builds (R5)."""
    from research.clfstudy.data import populate
    from research.outcomes import triggers as T

    T.ensure_builtins()
    return T.evaluate("gbb", populate(ohlcv[["date", *OHLCV]])).to_numpy(bool).astype(np.int8)


def _numeric(s: pd.Series) -> np.ndarray | None:
    if s.dtype == bool:
        return s.to_numpy(np.float64)
    if not pd.api.types.is_numeric_dtype(s):
        return None
    return s.to_numpy(np.float64, na_value=np.nan)


def _key(a: np.ndarray) -> str:
    b = np.where(np.isfinite(a), a, np.nan)
    return hashlib.sha1(np.ascontiguousarray(b).tobytes()).hexdigest()


def _defect(a: np.ndarray) -> str | None:
    fin = a[np.isfinite(a)]
    if len(fin) == 0:
        return "all_missing"
    if np.all(fin == fin[0]):
        return "constant"
    return None


def harvest_frame(ohlcv: pd.DataFrame, strats: list, aro, pair: str,
                  drop: set[str] = frozenset(), timing: dict | None = None,
                  keep_defects: bool = False):
    """(frame, info). `strats` = loaded in-scope instances; `aro` = the ARO control instance or
    None. `drop` = column names to exclude (causality FAILs). `keep_defects` keeps constant /
    all-missing columns (the truncation test compares every column, whatever its slice holds)."""
    cols: dict[str, np.ndarray] = {}
    info = {"classes": {}, "aliases": {}, "excluded_columns": {}}

    def put(name, a, source):
        if name in drop:
            info["excluded_columns"][name] = "lookahead"
            return
        bad = None if keep_defects else _defect(a)
        if bad:
            info["excluded_columns"][name] = bad
            return
        cols[name] = a

    t0 = time.perf_counter()
    base = populate_path(strats[0], ohlcv, pair)
    base_vals, base_keys = {}, {}
    for c in base.columns:
        if c in BASE_SKIP:
            continue
        a = _numeric(base[c])
        if a is None:
            continue
        base_vals[c] = a
        base_keys.setdefault(_key(a), f"base:{c}")
        put(f"base:{c}", a, "base")
    if timing is not None:
        timing["base"] = time.perf_counter() - t0

    for s in strats:
        name = type(s).__name__
        t0 = time.perf_counter()
        d = base.copy()
        d["atr_ok"] = True
        s.curr_pair = pair
        try:
            sig = O.raw_signals(s, d)
        except Exception as e:  # noqa: BLE001 -- recorded, the screen still runs (spec edge case)
            info["classes"][name] = {"status": "signal_error", "detail": repr(e)[:300]}
            continue
        harvested = []
        for c in d.columns:
            if c in BASE_SKIP:
                continue
            a = _numeric(d[c])
            if a is None:
                continue
            if c in base_vals and np.array_equal(a, base_vals[c], equal_nan=True):
                continue                                  # untouched base column
            full = f"{name}:{c}"
            k = _key(a)
            if k in base_keys:
                info["aliases"][full] = base_keys[k]
                continue
            put(full, a, name)
            harvested.append(full)
        for sk, a in sig.items():
            side, _, var = sk.partition(":")
            full = f"raw_{side}:{name}" + (f":{var}" if var else "")
            put(full, a.astype(np.float64), name)
            harvested.append(full)
        info["classes"][name] = {"status": "harvested", "override": O.kind(s),
                                 "columns": [c for c in harvested if c in cols]}
        if timing is not None:
            timing[name] = time.perf_counter() - t0

    if aro is not None:
        t0 = time.perf_counter()
        ad = populate_path(aro, ohlcv, pair)
        info["aro_include_list"] = list(aro.include_list)
        for c in aro.include_list:
            a = _numeric(ad[c])
            k = _key(a)
            if k in base_keys:
                info["aliases"][f"aro:{c}"] = base_keys[k]
            else:
                put(f"aro:{c}", a, "aro")
        if timing is not None:
            timing["aro"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    cols["base:gbb_candidate"] = gbb_candidate(ohlcv).astype(np.float64)
    if timing is not None:
        timing["gbb"] = time.perf_counter() - t0

    out = pd.DataFrame({"date": ohlcv["date"].to_numpy(), **{c: ohlcv[c].to_numpy() for c in OHLCV}})
    data = {}
    for c, a in cols.items():
        data[c] = a.astype(np.int8) if c.startswith("raw_") or c == "base:gbb_candidate" \
            else a.astype(np.float32)
    out = pd.concat([out, pd.DataFrame(data)], axis=1)
    return out, info


# ---------------------------------------------------------------------------- cache


def code_sha() -> str:
    h = hashlib.sha1()
    files = [HERE / "discover.py", HERE / "overrides.py", HERE / "harvest.py",
             *sorted((STRAT / "SimpleStrategies").glob("*.py"))]
    if CAUSALITY.exists():
        files.append(CAUSALITY)
    for f in files:
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def _slug(pair: str) -> str:
    return pair.replace("/", "_")


def cache_path(pair: str, cache: Path = CACHE) -> Path:
    return cache / f"{_slug(pair)}.feather"


def sidecar_path(pair: str, cache: Path = CACHE) -> Path:
    return cache / f"{_slug(pair)}.json"


def is_fresh(pair: str, sha: str, cache: Path = CACHE) -> bool:
    sc = sidecar_path(pair, cache)
    return cache_path(pair, cache).exists() and sc.exists() and \
        json.loads(sc.read_text()).get("code_sha") == sha


def write_cache(pair: str, frame: pd.DataFrame, info: dict, sha: str, cache: Path = CACHE) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    frame.to_feather(cache_path(pair, cache))
    sidecar_path(pair, cache).write_text(json.dumps({
        "code_sha": sha, "pair": pair, "rows": len(frame),
        "first_date": str(frame["date"].iloc[0]), "last_date": str(frame["date"].iloc[-1]),
        "columns": [c for c in frame.columns if c not in ("date", *OHLCV)], **info}, indent=1))


def read_cache(pair: str, columns: list[str] | None = None, cache: Path = CACHE) -> pd.DataFrame:
    """Read selected columns (date always included) of one pair's cache."""
    p = cache_path(pair, cache)
    if columns is None:
        return pd.read_feather(p)
    have = set(json.loads(sidecar_path(pair, cache).read_text())["columns"]) | {"date", *OHLCV}
    return pd.read_feather(p, columns=["date", *[c for c in columns if c in have and c != "date"]])


def cache_columns(pair: str, cache: Path = CACHE) -> list[str]:
    return json.loads(sidecar_path(pair, cache).read_text())["columns"]


def causality_fails() -> set[str]:
    if not CAUSALITY.exists():
        return set()
    return {c for c, r in json.loads(CAUSALITY.read_text())["columns"].items() if r["verdict"] == "FAIL"}


def _override(per_pair: dict, cls: str) -> str | None:
    for v in per_pair.values():
        k = v["classes"].get(cls, {}).get("override")
        if k:
            return None if k == "plain" else k
    return None


def write_manifest(scope: dict, per_pair: dict, sha: str, timing: dict | None = None) -> None:
    caus = json.loads(CAUSALITY.read_text())["columns"] if CAUSALITY.exists() else {}
    names = sorted({c for v in per_pair.values() for c in v["columns"]})
    columns = []
    for c in names:
        src = c.split(":")[1] if c.startswith("raw_") else c.split(":")[0]
        k = "raw_buy" if c.startswith("raw_buy:") else "raw_sell" if c.startswith("raw_sell:") \
            else "candidate" if c == "base:gbb_candidate" else "meta" if c.startswith("meta:") \
            else "value"
        r = caus.get(c, {})
        e = {"name": c, "source": src, "kind": k, "causality": r.get("verdict", "UNTESTED")}
        if r.get("first_mismatch"):
            e["first_mismatch"] = r["first_mismatch"]
        columns.append(e)
    excluded = list(scope["excluded"])
    for pair, v in sorted(per_pair.items()):
        for cls, ci in v["classes"].items():
            if ci["status"] != "harvested":
                excluded.append({"class": cls, "reason": ci["status"], "pair": pair,
                                 "detail": ci.get("detail")})
    old = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    MANIFEST.write_text(json.dumps({
        "code_sha": sha,
        "classes": [{**c, "override": _override(per_pair, c["class"])} for c in scope["classes"]],
        "excluded": excluded,
        "excluded_columns": {p: v["excluded_columns"] for p, v in sorted(per_pair.items())},
        "aliases": next(iter(per_pair.values()))["aliases"] if per_pair else {},
        "columns": columns,
        "pairs": sorted(per_pair),
        "range": {"start": W.DATA_START, "end": W.DATA_END, "warmup_bars": W.WARMUP_BARS},
        "timing": timing if timing is not None else old.get("timing"),
    }, indent=1))


# ---------------------------------------------------------------------------- CLI


def _busy() -> list[str]:
    from research.candidate_training import train as drv

    return drv.other_chains(drv._ps(), os.getpid())


def load_scope(cfg):
    scope = D.discover()
    strats = [load_strategy(cfg, c["class"]) for c in scope["classes"]]
    return scope, strats


def run_pairs(pair_list: list[str], force: bool = False) -> int:
    _quiet()
    cfg = base_config()
    scope, strats = load_scope(cfg)
    aro = load_aro()
    sha = code_sha()
    drop = causality_fails()
    per_pair = {}
    for pair in pair_list:
        if not force and is_fresh(pair, sha):
            print(f"{pair}: cache fresh ({sha})")
            per_pair[pair] = json.loads(sidecar_path(pair).read_text())
            continue
        t0 = time.perf_counter()
        ohlcv = load_ohlcv(cfg, pair)
        frame, info = harvest_frame(ohlcv, strats, aro, pair, drop)
        frame["meta:has_data"], frame["meta:data_ok"] = data_flags(
            ohlcv["date"], load_ohlcv(cfg, pair, filled=False)["date"])
        write_cache(pair, frame, info, sha)
        per_pair[pair] = {**info, "columns": [c for c in frame.columns if c not in ("date", *OHLCV)]}
        print(f"{pair}: {len(frame)} rows, {frame.shape[1]} columns, "
              f"{time.perf_counter() - t0:.0f}s", flush=True)
    write_manifest(scope, per_pair, sha)
    return 0


def time_one(pair: str) -> int:
    _quiet()
    cfg = base_config()
    scope, strats = load_scope(cfg)
    aro = load_aro()
    timing = {}
    t0 = time.perf_counter()
    frame, info = harvest_frame(load_ohlcv(cfg, pair), strats, aro, pair, timing=timing)
    total = time.perf_counter() - t0
    for k, v in sorted(timing.items(), key=lambda kv: -kv[1])[:15]:
        print(f"  {k:<20} {v:7.1f}s")
    print(f"{pair}: {frame.shape[1]} columns, total {total:.0f}s; 11 frames ~ {11 * total / 60:.0f} min")
    m = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    m["timing"] = {"pair": pair, "total_s": round(total, 1), "estimate_11_frames_min":
                   round(11 * total / 60, 1), "per_class_s": {k: round(v, 2) for k, v in timing.items()}}
    MANIFEST.write_text(json.dumps(m, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pair")
    g.add_argument("--time-one", metavar="PAIR")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    busy = _busy()
    if busy:
        print(f"REFUSED: a training/backtest chain is running: {busy}")
        return 1
    if a.time_one:
        return time_one(a.time_one)
    return run_pairs(pairs() if a.pair == "all" else [a.pair], a.force)


if __name__ == "__main__":
    raise SystemExit(main())
