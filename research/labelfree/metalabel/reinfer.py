"""Recover entry-time features for every archived trade (T009-T012, FR-005a/b, R1, R8).

MECHANISM. freqtrade's own `--export signals` stores, per pair, the last candle STRICTLY
BEFORE each trade's open_date:

    candle_idx = pairdf["date"].searchsorted(pairresults[date_col], side="left") - 1

(`optimize_reports.generate_trade_signal_candles`). Those rows carry every dataframe column,
including the `%predict_prob_{sell,hold,buy}` columns the framework now writes. So the
entry-time feature table is produced by freqtrade itself, keyed to the trades that actually
happened, with causality guaranteed by construction rather than by our own bookkeeping.

REPRODUCTION GATE (FR-005b, SC-017). The re-inferred (pair, open_date) set must equal the
archived set EXACTLY. The archived trades were produced 2026-09-09; these features come from
re-running those models now. If the sets differ, the features describe a different model than
the one whose outcomes are the training target, and every downstream number is void. A
mismatch halts the feature -- it means the fold models, their pinned params, or the feather
data have changed, each of which is a finding in its own right.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
REPO = ROOT.parents[1]                       # the directory containing user_data/
sys.path.insert(0, str(ROOT))

from research.labelfree.metalabel import archive as A    # noqa: E402
from research.labelfree import metalabel as M            # noqa: E402

PROB_COLS = ("%predict_prob_sell", "%predict_prob_hold", "%predict_prob_buy")
TIMEFRAME_MIN = 15
CONFIG = "user_data/strategies/config/config_wf.json"


def model_store(fold_id: str) -> Path:
    return ROOT / M.MODEL_STORE / f"NNNC_WF_{fold_id}"


def require_models(fold_ids) -> None:
    """Refuse to run on a missing fold model rather than silently training on a subset."""
    missing = [f for f in fold_ids if not model_store(f).is_dir()]
    if missing:
        raise SystemExit(
            f"missing per-fold model stores: {missing}. The feature depends on "
            f"{M.MODEL_STORE}/NNNC_WF_* surviving; a silent skip would train the secondary "
            "model on a subset of folds without saying so.")


def run_fold(fold_id: str, fold: dict, log_dir: Path) -> Path:
    """Backtest one fold over its own test range with signal export. Returns the result zip.

    TWO TRAPS, both measured 2026-09-12:

    1. FREQTRADE EXITS 0 ON A FATAL CONFIG ERROR. `rc != 0` is not a failure signal --
       a run that died on config validation returns 0 and writes nothing. Freshness of the
       result file is the only trustworthy check, so the zip's mtime must postdate the run.
       Without this a failed fold silently yields the PREVIOUS run's results and the whole
       feature analyses the wrong data.
    2. Secrets live in ~/.freqtrade_env, sourced by ~/.zshrc for INTERACTIVE shells only.
       A non-interactive `zsh -c` does not get them, and config validation then fails on
       jwt_secret_key. It is sourced explicitly here.
    """
    tr = f"{fold['test_start'].replace('-', '')}-{fold['test_end'].replace('-', '')}"
    log = log_dir / f"reinfer_{fold_id}.log"
    results_dir = REPO / "user_data/backtest_results"
    started = time.time()
    cmd = ("[ -f \"$HOME/.freqtrade_env\" ] && . \"$HOME/.freqtrade_env\"; "
           f"cd {REPO} && PYTHONPATH=user_data/strategies .venv/bin/freqtrade backtesting "
           f"--strategy NNNC_WF_{fold_id} --config {CONFIG} --timerange {tr} "
           f"--export signals --cache none")
    with log.open("w") as fh:
        subprocess.run(["zsh", "-c", cmd], stdout=fh, stderr=fh)

    latest = results_dir / ".last_result.json"
    if not latest.exists():
        raise SystemExit(f"{fold_id}: no .last_result.json; see {log}")
    zp = results_dir / json.loads(latest.read_text())["latest_backtest"]
    if not zp.exists():
        raise SystemExit(f"{fold_id}: {zp.name} missing; see {log}")
    if zp.stat().st_mtime < started:
        raise SystemExit(
            f"{fold_id}: STALE RESULT. {zp.name} predates this run, so the backtest "
            f"produced nothing -- freqtrade exits 0 even on a fatal config error, so this "
            f"is the only reliable failure signal. See {log}")
    return zp


def read_signals(zip_path: Path) -> dict[str, pd.DataFrame]:
    """The `<base>_signals.pkl` joblib payload: {pair: DataFrame of entry candles}."""
    import joblib
    with ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.endswith("_signals.pkl")]
        if not names:
            raise SystemExit(
                f"{zip_path.name} carries no _signals.pkl. The backtest must run with "
                "`--export signals`; without it no entry-time features exist.")
        with zf.open(names[0]) as fh:
            return joblib.load(fh)


def join_fold(rows: list[A.TradeRow], signals: dict[str, pd.DataFrame]) -> tuple[list, list]:
    """Attach each trade's entry-candle features. Returns (built, excluded).

    Joined on the candle DATE, not on row order: `generate_trade_signal_candles` drops trades
    whose candle index is negative, so positional alignment silently shifts every later row
    for that pair. The signal candle is the bar immediately before the entry, so its date is
    `open_date - timeframe`.
    """
    built, excluded = [], []
    for r in rows:
        df = signals.get(r.pair)
        if df is None or df.empty:
            excluded.append((r, "no signal candles for pair")); continue
        want = pd.Timestamp(r.open_date) - pd.Timedelta(minutes=TIMEFRAME_MIN)
        hit = df[df["date"] == want]
        if hit.empty:
            excluded.append((r, "no candle at open_date - timeframe")); continue
        row = hit.iloc[0]
        if any(pd.isna(row.get(c)) for c in PROB_COLS):
            # NaN marks a bar the model never scored (the front-padded warm-up region).
            # Excluded and counted, never imputed (FR-006) -- and never zero-filled, since
            # 0.0 is a real probability meaning "confidently not this class".
            excluded.append((r, "nan_confidence")); continue
        built.append((r, row))
    return built, excluded


def reproduction_gate(archived: set, reinferred: set) -> dict:
    """FR-005b / SC-017. Exact set equality, or the feature halts."""
    missing, extra = sorted(archived - reinferred), sorted(reinferred - archived)
    return {"matches_exactly": not missing and not extra,
            "n_archived": len(archived), "n_reinferred": len(reinferred),
            "missing": missing[:20], "extra": extra[:20],
            "n_missing": len(missing), "n_extra": len(extra)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive", default=str(ROOT / M.ARCHIVE_PATH))
    ap.add_argument("--folds", help="comma-separated subset; default all 15")
    ap.add_argument("--out", required=True)
    ap.add_argument("--log-dir", default="/tmp/lf")
    ap.add_argument("--from-zips", help="skip backtests; read these result zips (comma-sep)")
    a = ap.parse_args()

    rows = A.load(a.archive)
    plan = A.fold_ranges()
    fold_ids = a.folds.split(",") if a.folds else sorted(plan)
    require_models(fold_ids)
    log_dir = Path(a.log_dir); log_dir.mkdir(parents=True, exist_ok=True)

    by_fold = A.by_fold(rows)
    zips = a.from_zips.split(",") if a.from_zips else []
    built_all, excluded_all, records = [], [], []

    for i, fid in enumerate(fold_ids):
        frows = by_fold.get(fid, [])
        zp = Path(zips[i]) if zips else run_fold(fid, plan[fid], log_dir)
        sig = read_signals(zp)
        built, excluded = join_fold(frows, sig)
        gate = reproduction_gate({r.key for r in frows}, {r.key for r, _ in built})
        gate["fold_id"] = fid
        gate["n_excluded"] = len(excluded)
        records.append(gate)
        print(f"  {fid}: archived={len(frows)} built={len(built)} excluded={len(excluded)} "
              f"matches_exactly={gate['matches_exactly']}", flush=True)
        built_all += built; excluded_all += excluded

    bad = [r for r in records if not r["matches_exactly"]]
    if bad:
        print("\n*** REPRODUCTION GATE FAILED (FR-005b, SC-017) ***")
        for r in bad:
            print(f"  {r['fold_id']}: {r['n_missing']} missing, {r['n_extra']} extra")
            for k in r["missing"][:5]:
                print(f"    missing {k}")
        print("\nNo output written. The features would describe a different model than the "
              "one whose outcomes are the training target.")
        return 1

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for r, row in built_all:
            d = r.describe()
            d["features"] = {c: float(row[c]) for c in PROB_COLS if c in row}
            fh.write(json.dumps(d) + "\n")
    print(f"\n  wrote {len(built_all)} rows -> {out}")
    print(f"  excluded {len(excluded_all)} (counted, never imputed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
