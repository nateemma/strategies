"""Backtest-arm runner for the market-phase study (contracts/study-cli.md, T006).

Runs one arm of a comparison at a pinned window with a chosen entry threshold,
exports its realized trades, and writes a MechanismArm record.

WHY THIS EXISTS RATHER THAN test_strat.sh:
  - test_strat.sh has no --export, so realized trades are not written out.
  - Its -n/-o options compute the window RELATIVE TO TODAY, so two arms run on
    different days silently cover different periods (research R8).
  - Modifying it would change behaviour for every other experiment in the repo.

HOW PER-ARM PARAMETERS ARE APPLIED (research R5, verified against
freqtrade/strategy/hyper.py:load_params_from_file):

  freqtrade derives the parameter file from the strategy MODULE'S OWN __file__
  with a .json suffix, and validates params["strategy_name"] against the class
  name. It then does deep_merge_dicts(file_params, class_params) where the FILE
  wins. So NNNC/NNNC_MLX.json overrides anything set in Python -- setting a
  threshold in the class silently does nothing.

  This runner therefore builds a scratch strategy directory containing SYMLINKS
  to every file in NNNC/ plus a REAL per-arm NNNC_MLX.json, and points
  --strategy-path at it. Consequences:
    - the class name stays NNNC_MLX, so get_model_path() resolves to the SAME
      trained model and no arm triggers a retrain;
    - the user's NNNC/NNNC_MLX.json is never read, written, or moved.

Run from the freqtrade root:

    PYTHONPATH=. .venv/bin/python user_data/strategies/regime/run_arm.py \
        --strategy NNNC_MLX --window W1 --threshold 0.69 --tag baseline
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

from regime.regimedata import WINDOWS

ROOT = Path("/Users/philprice95/projects/freqtrade")
STRAT_DIR = ROOT / "user_data/strategies"
OUT_DIR = STRAT_DIR / "regime/arms"
SCRATCH = STRAT_DIR / "regime/.scratch"
DEFAULT_CONFIG = STRAT_DIR / "config/config.json"

# Which subdirectory of user_data/strategies holds the host strategy.
STRATEGY_GROUP = {"NNNC_MLX": "NNNC"}


def timerange(window: str) -> str:
    """Resolve a named window to a LITERAL freqtrade timerange.

    Never today-relative: an arm run on a different day must cover the same
    period or it is not comparable (FR-026, research R8).
    """
    start, end = WINDOWS[window]
    return f"{start}-{end}"


def build_scratch(strategy: str, threshold: float, tag: str, window: str,
                  guards: bool = True, overrides: dict | None = None) -> Path:
    """Symlink the strategy group into a scratch dir and write the arm's JSON.

    Returns the scratch directory to pass as --strategy-path.
    """
    group = STRATEGY_GROUP[strategy]
    src = STRAT_DIR / group
    dst = SCRATCH / f"{tag}_{window}"
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    # Symlink every module so the strategy's own
    # `sys.path.append(Path(__file__).parent)` still resolves its siblings.
    for f in src.glob("*.py"):
        (dst / f.name).symlink_to(f.resolve())

    user_json = src / f"{strategy}.json"
    if not user_json.is_file():
        raise SystemExit(f"expected {user_json} to exist - it carries the tuned params")
    params = json.loads(user_json.read_text())

    buy = params.setdefault("params", {}).setdefault("buy", {})
    buy["prediction_threshold"] = threshold
    # The guard chain is ANDed with the model condition. With it ON and tuned
    # hard (rvol>3.1, adx>38, guard_metric<-0.4) the guards can be the binding
    # constraint, leaving the z-gate almost no room to vary - the "judge at a
    # powered operating point" trap. Allow measuring with it off.
    if not guards:
        buy["entry_enable_guards"] = False
    # Arbitrary buy-space overrides, for sweeping individual guards.
    for k, v in (overrides or {}).items():
        buy[k] = v
    (dst / f"{strategy}.json").write_text(json.dumps(params, indent=2))

    return dst


def exit_config_hash(scratch: Path, strategy: str) -> str:
    """Hash of the arm's exit-side configuration.

    Must match the baseline's for the same window, proving no arm altered exit
    behaviour (FR-024, SC-010).
    """
    params = json.loads((scratch / f"{strategy}.json").read_text()).get("params", {})
    exit_side = {k: params.get(k) for k in ("sell", "roi", "stoploss", "trailing")}
    blob = json.dumps(exit_side, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def build_env(scratch: Path) -> dict:
    """Environment for the backtest subprocess.

    The strategy imports `Framework` and `utils` as top-level packages, which
    only resolve with user_data/strategies on PYTHONPATH - test_strat.sh sets
    exactly this (line 179) and it is easy to miss when invoking freqtrade
    directly. Without it the strategy fails to import and freqtrade reports
    only "This class does not exist or contains Python code errors".
    """
    env = os.environ.copy()
    parts = [".", str(STRAT_DIR), str(scratch), str(STRAT_DIR / "utils")]
    env["PYTHONPATH"] = ":".join(parts + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    return env


def build_command(strategy: str, scratch: Path, window: str, config: Path,
                  export_path: Path) -> list[str]:
    return [
        str(ROOT / ".venv/bin/freqtrade"), "backtesting",
        "--strategy", strategy,
        "--strategy-path", str(scratch),
        "-c", str(config),
        "--timerange", timerange(window),
        "--cache", "none",
        "--export", "trades",
        "--breakdown", "month",
    ]


BT_RESULTS = ROOT / "user_data/backtest_results"


def latest_result() -> Path | None:
    """Path of the most recent backtest archive.

    `--export-filename` is DEPRECATED and SILENTLY IGNORED when backtesting -
    freqtrade logs "DEPRECATED: Using `--export-filename` has no impact when
    backtesting" and writes to user_data/backtest_results/ regardless. Reading
    the requested path instead reports every arm as zero-trade/inconclusive,
    which is a false null of exactly the shape Principle IV forbids. This is the
    SIXTH recorded instance of a parameter accepted, logged, and ignored.
    """
    marker = BT_RESULTS / ".last_result.json"
    if not marker.is_file():
        return None
    name = json.loads(marker.read_text()).get("latest_backtest")
    p = BT_RESULTS / name if name else None
    return p if p and p.exists() else None


def summarize(result_path: Path | None) -> dict:
    """Derive the arm's measured outcome from the exported trades.

    LIVENESS (FR-031, Constitution IV): rather than instrumenting the strategy
    to count entry-chain passes - which would mean editing Framework/, and
    Phases A-C touch no strategy code - inertness is detected by hashing the
    realized trade list. Two arms whose trade lists hash identically produced a
    BYTE-IDENTICAL backtest, which is exactly the tell recorded for the known
    silently-inert parameters, and stronger than one pass count.
    """
    from freqtrade.data.btanalysis import load_backtest_data, load_backtest_stats

    if result_path is None:
        return {"n_trades": 0, "trades_hash": None, "note": "NO RESULT FOUND - do not read as zero trades"}

    trades = load_backtest_data(result_path)
    if trades is None or trades.empty:
        return {"n_trades": 0, "trades_hash": None, "note": "zero trades - INCONCLUSIVE"}

    cols = [c for c in ("pair", "open_date", "close_date", "profit_ratio") if c in trades]
    blob = trades[cols].to_csv(index=False).encode()
    dur_h = (trades["close_date"] - trades["open_date"]).dt.total_seconds() / 3600.0
    return {
        "result_file": result_path.name,
        "n_trades": int(len(trades)),
        "n_pairs": int(trades["pair"].nunique()),
        "trades_hash": hashlib.sha256(blob).hexdigest()[:16],
        "profit_pct": round(float(trades["profit_ratio"].sum() * 100), 3),
        "win_rate": round(float((trades["profit_ratio"] > 0).mean() * 100), 1),
        "duration_h_median": round(float(dur_h.median()), 2),
        "duration_h_p25": round(float(dur_h.quantile(0.25)), 2),
        "duration_h_p75": round(float(dur_h.quantile(0.75)), 2),
        "trades_per_day": round(len(trades) / max((trades["close_date"].max() - trades["open_date"].min()).days, 1), 4),
        **_wallet_stats(result_path),
    }


def _wallet_stats(result_path: Path) -> dict:
    """Wallet-level outcome. Summed profit_ratio is NOT the portfolio return -
    with hundreds of trades it vastly overstates it."""
    try:
        from freqtrade.data.btanalysis import load_backtest_stats
        st = load_backtest_stats(result_path)
        s = list(st["strategy"].values())[0]
        return {"wallet_profit_pct": round(float(s.get("profit_total", 0)) * 100, 3),
                "max_drawdown_pct": round(float(s.get("max_drawdown_account", 0)) * 100, 3),
                "profit_factor": round(float(s.get("profit_factor", 0)), 3)}
    except Exception as e:
        return {"wallet_profit_pct": None, "max_drawdown_pct": None,
                "profit_factor": None, "stats_error": str(e)[:80]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strategy", default="NNNC_MLX",
                    help="host strategy; never an NNNC_DDPM_ variant (FR-033)")
    ap.add_argument("--window", required=True, choices=sorted(WINDOWS))
    ap.add_argument("--threshold", type=float, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a buy-space parameter, repeatable "
                         "(e.g. --set entry_rvol_threshold=0.5)")
    ap.add_argument("--guards", choices=["on", "off"], default="on",
                    help="entry_enable_guards; 'off' loosens the gate so the "
                         "model threshold has room to bite")
    ap.add_argument("--plan", action="store_true",
                    help="build the scratch dir and print the command without running it")
    args = ap.parse_args()

    if args.strategy.startswith("NNNC_DDPM_"):
        raise SystemExit("NNNC_DDPM_ prefix activates a GAN; this feature uses the non-GAN base")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    overrides = {}
    for kv in args.set:
        k, _, v = kv.partition("=")
        overrides[k] = float(v) if v.replace(".", "").replace("-", "").isdigit() else v
    scratch = build_scratch(args.strategy, args.threshold, args.tag, args.window,
                            guards=(args.guards == "on"), overrides=overrides)
    export_path = OUT_DIR / f"{args.tag}_{args.window}_trades.json"
    cmd = build_command(args.strategy, scratch, args.window, args.config, export_path)

    print(f"arm       : {args.tag} / {args.window}")
    print(f"timerange : {timerange(args.window)}  (literal, not today-relative)")
    print(f"threshold : {args.threshold}")
    print(f"scratch   : {scratch}")
    print(f"exit hash : {exit_config_hash(scratch, args.strategy)}")
    print(f"command   : {' '.join(cmd)}")

    if args.plan:
        print("\n--plan: not executed.")
        return 0

    proc = subprocess.run(cmd, cwd=ROOT, env=build_env(scratch))
    result = summarize(latest_result())

    record = {
        "tag": args.tag,
        "window": args.window,
        "timerange": timerange(args.window),
        "strategy": args.strategy,
        "prediction_threshold": args.threshold,
        "guards": args.guards,
        "overrides": overrides,
        "exit_config_hash": exit_config_hash(scratch, args.strategy),
        "volume_filter_active": True,
        "returncode": proc.returncode,
        **result,
    }
    out = OUT_DIR / f"{args.tag}_{args.window}.json"
    out.write_text(json.dumps(record, indent=2))
    print(f"\nwrote {out}")
    print(json.dumps(record, indent=2))
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
