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
import re
import shutil
import subprocess
import time
import sys
from pathlib import Path

import pandas as pd

from regime.regimedata import WINDOWS
from utils.ExitPolicy import SIMULATION_LOGIC_VERSION

ROOT = Path("/Users/philprice95/projects/freqtrade")
STRAT_DIR = ROOT / "user_data/strategies"
OUT_DIR = STRAT_DIR / "regime/arms"
SCRATCH = STRAT_DIR / "regime/.scratch"
DEFAULT_CONFIG = STRAT_DIR / "config/config.json"

# Which subdirectory of user_data/strategies holds the host strategy.
STRATEGY_GROUP = {"NNNC_MLX": "NNNC", "NNMT_MLX": "NNMT"}


def strategy_group(strategy: str) -> str:
    """Directory holding `strategy`, resolving family variants by prefix.

    Ablation and variant classes (NNMT_MLX_noflow, NNMT_MLX_ctrl, ...) inherit
    from a family base and live beside it in the same directory, so an exact
    lookup would reject them. Longest prefix wins, so adding a more specific
    base later cannot be shadowed by a shorter one.
    """
    if strategy in STRATEGY_GROUP:
        return STRATEGY_GROUP[strategy]
    for base in sorted(STRATEGY_GROUP, key=len, reverse=True):
        if strategy.startswith(base):
            return STRATEGY_GROUP[base]
    # Family variants that do not extend a registered base's full name --
    # e.g. NNNC_BreakoutOpen, which is an NNNC strategy but not an NNNC_MLX one.
    # Fall back to the leading family token when it names a real directory.
    family = strategy.split("_", 1)[0]
    if (STRAT_DIR / family).is_dir():
        return family
    # Last resort, added 2026-10-09: find the file. A strategy whose name carries no family
    # token -- a plain `SimpleStrategies/` strategy such as `BreakoutVolOpen`, with no
    # underscore to split on -- reaches here even though its file is sitting in a strategy
    # directory. This branch is ADDITIVE: it runs only where the code above would already
    # have raised, so it cannot change how any existing arm resolves. Deliberately exact on
    # the filename, and ambiguity is an error rather than a guess.
    matches = sorted(
        d.name for d in STRAT_DIR.iterdir()
        if d.is_dir() and (d / f"{strategy}.py").is_file()
    )
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise SystemExit(
            f"ambiguous strategy {strategy!r}: {strategy}.py exists in {matches}. "
            f"Rename one, or pass the family explicitly -- guessing would silently run the "
            f"wrong file."
        )
    raise SystemExit(
        f"unknown strategy {strategy!r}: no registered base matches and "
        f"{family!r} is not a strategy directory. Known bases: "
        f"{sorted(STRATEGY_GROUP)}"
    )


def timerange(window: str) -> str:
    """Resolve a named window to a LITERAL freqtrade timerange.

    Never today-relative: an arm run on a different day must cover the same
    period or it is not comparable (FR-026, research R8).
    """
    start, end = WINDOWS[window]
    return f"{start}-{end}"


def resolve_window(window: str | None, tr: str | None,
                   label: str | None) -> tuple[str, str]:
    """(literal timerange, naming label) from either --window or --timerange.

    `--window` names a period in `regime.regimedata.WINDOWS`; `--timerange` gives one
    literally. The two are alternatives, and exactly one must be supplied.

    WHY --timerange EXISTS. The pooled walk-forward harness runs ~15 arbitrary fold ranges
    that are not named windows. Before this, it had to inject a fold into `rd.WINDOWS` in an
    in-memory subprocess to get a range past `--window`'s `choices=`. That hack was the single
    largest thing welding the harness to `regime/`, and it is why this argument was worth
    adding on its own merits. Do NOT instead add fold keys to WINDOWS on disk --
    `phase_c_trackability.py` iterates that dict.
    """
    if (window is None) == (tr is None):
        raise SystemExit("give exactly one of --window or --timerange")
    if window is not None:
        return timerange(window), window
    if not re.fullmatch(r"\d{8}-\d{8}", tr or ""):
        raise SystemExit(f"--timerange must be YYYYMMDD-YYYYMMDD, got {tr!r}")
    return tr, (label or tr)


def build_scratch(strategy: str, threshold: float, tag: str, window: str,
                  guards: bool = True, overrides: dict | None = None,
                  sell_overrides: dict | None = None, roi: dict | None = None,
                  stoploss: float | None = None, trailing: dict | None = None) -> Path:
    """Symlink the strategy group into a scratch dir and write the arm's JSON.

    Returns the scratch directory to pass as --strategy-path.
    """
    group = strategy_group(strategy)
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
    if user_json.is_file():
        params = json.loads(user_json.read_text())
    else:
        # A strategy with no params file runs purely on its class defaults.
        # Synthesize the minimum freqtrade needs so overrides can still apply;
        # load_params_from_file validates strategy_name against the class.
        params = {"strategy_name": strategy, "params": {}}

    buy = params.setdefault("params", {}).setdefault("buy", {})
    if threshold is not None:
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
    # Sell space, ROI table and stoploss live OUTSIDE params.buy. Pinning them
    # explicitly matters: the user's NNNC_MLX.json has already drifted mid-study
    # (a hyperopt overwrote the exit block), so an arm that inherits it silently
    # measures a different baseline (FR-028b).
    sell = params.setdefault("params", {}).setdefault("sell", {})
    for k, v in (sell_overrides or {}).items():
        sell[k] = v
    if roi is not None:
        params.setdefault("params", {})["roi"] = roi
    if stoploss is not None:
        params.setdefault("params", {})["stoploss"] = {"stoploss": stoploss}
    if trailing:
        params.setdefault("params", {}).setdefault("trailing", {}).update(trailing)
    (dst / f"{strategy}.json").write_text(json.dumps(params, indent=2))

    return dst


def guards_written(scratch: Path, strategy: str) -> "bool | None":
    """The `entry_enable_guards` value this run actually WROTE, or None if it wrote none.

    Reads the params file on disk rather than trusting the in-memory dict, for the same
    reason `exit_config_hash` does: what the backtest saw is what was written. None means
    this run did not override the setting, so the effective value is the strategy class's
    own `buy_params` -- which this script deliberately does not import, so it reports the
    absence instead of guessing.
    """
    try:
        params = json.loads((scratch / f"{strategy}.json").read_text()).get("params", {})
        return params.get("buy", {}).get("entry_enable_guards")
    except (OSError, ValueError):
        return None


def exit_config_hash(scratch: Path, strategy: str) -> str:
    """Hash of the arm's exit-side configuration.

    Must match the baseline's for the same window, proving no arm altered exit
    behaviour (FR-024, SC-010).
    """
    params = json.loads((scratch / f"{strategy}.json").read_text()).get("params", {})
    exit_side = {k: params.get(k) for k in ("sell", "roi", "stoploss", "trailing")}
    # SIMULATION_LOGIC_VERSION folded in 2026-10-08, mirroring
    # `research.outcomes.table.ExitConfig.config_hash`. Hashing the params ALONE made this
    # blind to a change in the exit CODE: the post-grace stop sign fix altered realised
    # exits on 191 of 1420 trades and every arm record still read the same
    # c8f80f6af046d088, so "no arm altered exit behaviour" could be satisfied by two runs
    # whose exits genuinely differed. The invariant this hash exists to prove is unchanged
    # and now strictly stronger: all arms in one run still share a version, so a mismatch
    # between arms still means a PARAMS divergence, while a mismatch against an older
    # record now correctly flags that the arithmetic moved underneath it.
    blob = json.dumps({**exit_side, "simulation_logic_version": SIMULATION_LOGIC_VERSION},
                      sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _parse_roi(v: str) -> dict:
    """Accept either a flat level or a full ROI ladder as JSON."""
    v = v.strip()
    if v.startswith("{"):
        return {str(k): float(x) for k, x in json.loads(v).items()}
    return {"0": float(v)}


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


def build_command(strategy: str, scratch: Path, tr: str, config: Path,
                  export_path: Path) -> list[str]:
    return [
        str(ROOT / ".venv/bin/freqtrade"), "backtesting",
        "--strategy", strategy,
        "--strategy-path", str(scratch),
        "-c", str(config),
        "--timerange", tr,
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
                "profit_factor": round(float(s.get("profit_factor", 0)), 3),
                "calmar": round(float(s.get("calmar", 0)), 3),
                "sortino": round(float(s.get("sortino", 0)), 3),
                "sqn": round(float(s.get("sqn", 0)), 3),
                "expectancy_ratio": round(float(s.get("expectancy_ratio", 0)), 4)}
    except Exception as e:
        return {"wallet_profit_pct": None, "max_drawdown_pct": None,
                "profit_factor": None, "stats_error": str(e)[:80]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strategy", default="NNNC_MLX",
                    help="host strategy; never an NNNC_DDPM_ variant (FR-033)")
    ap.add_argument("--window", choices=sorted(WINDOWS),
                    help="named period from regime.regimedata.WINDOWS")
    ap.add_argument("--timerange", default=None, metavar="YYYYMMDD-YYYYMMDD",
                    help="literal range; alternative to --window, for arbitrary fold ranges")
    ap.add_argument("--label", default=None,
                    help="naming slot for --timerange runs (default: the timerange itself)")
    ap.add_argument("--threshold", type=float, default=None,
                    help="prediction_threshold; omit to keep the strategy's own default")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--set-sell", action="append", default=[], metavar="KEY=VALUE",
                    help="override a sell-space parameter, repeatable")
    ap.add_argument("--roi", default=None, metavar="PCT|JSON",
                    help='flat ROI level (e.g. 0.03) or a full ladder as JSON '
                         '(e.g. \'{"0":0.06,"180":0.04}\')')
    ap.add_argument("--stoploss", type=float, default=None,
                    help="Static stoploss. WARNING: INERT for every NN strategy. "
                    "BaseNNStrategy sets use_atr_adaptive_stoploss=True, so "
                    "_resolve_normal_stop() returns an ATR-derived value and ignores "
                    "self.stoploss; this flag only changes the cosmetic "
                    "initial_stop_loss_abs. To move the real stop, override "
                    "atr_stoploss_multiplier/floor/cap on a subclass.")
    ap.add_argument("--set-trailing", action="append", default=[], metavar="KEY=VALUE",
                    help="override a trailing-block parameter, repeatable")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a buy-space parameter, repeatable "
                         "(e.g. --set entry_rvol_threshold=0.5)")
    ap.add_argument("--guards", choices=["on", "off"], default="on",
                    help="entry_enable_guards; 'off' loosens the gate so the "
                         "model threshold has room to bite")
    ap.add_argument("--allow-gan", action="store_true",
                    help="permit a GAN-activating strategy (NNNC_DDPM_*). The "
                         "guard exists so the regime study cannot silently run a "
                         "GAN variant when the non-GAN base was meant; set this "
                         "only when comparing GAN arms deliberately.")
    ap.add_argument("--plan", action="store_true",
                    help="build the scratch dir and print the command without running it")
    args = ap.parse_args()

    if args.strategy.startswith("NNNC_DDPM_") and not args.allow_gan:
        raise SystemExit(
            "NNNC_DDPM_ prefix activates a GAN and the regime study uses the "
            "non-GAN base. Pass --allow-gan if a GAN arm is genuinely intended."
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    overrides = {}
    for kv in args.set:
        k, _, v = kv.partition("=")
        overrides[k] = float(v) if v.replace(".", "").replace("-", "").isdigit() else v
    sell_over = {}
    for kv in args.set_sell:
        k, _, v = kv.partition("=")
        if v.lower() in ("true", "false"):
            sell_over[k] = v.lower() == "true"
        else:
            try: sell_over[k] = float(v)
            except ValueError: sell_over[k] = v
    trail_over = {}
    for kv in args.set_trailing:
        k, _, v = kv.partition("=")
        trail_over[k] = (v.lower() == "true") if v.lower() in ("true","false") else float(v)
    tr, label = resolve_window(args.window, args.timerange, args.label)
    scratch = build_scratch(args.strategy, args.threshold, args.tag, label,
                            guards=(args.guards == "on"), overrides=overrides,
                            sell_overrides=sell_over,
                            roi=(_parse_roi(args.roi) if args.roi else None),
                            stoploss=args.stoploss, trailing=trail_over)
    export_path = OUT_DIR / f"{args.tag}_{label}_trades.json"
    cmd = build_command(args.strategy, scratch, tr, args.config, export_path)

    print(f"arm       : {args.tag} / {label}")
    print(f"timerange : {tr}  (literal, not today-relative)")
    print(f"threshold : {args.threshold if args.threshold is not None else '(strategy default)'}")
    print(f"scratch   : {scratch}")
    print(f"exit hash : {exit_config_hash(scratch, args.strategy)}")
    print(f"command   : {' '.join(cmd)}")

    if args.plan:
        print("\n--plan: not executed.")
        return 0

    started = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, env=build_env(scratch))

    # A FAILED backtest must never yield metrics. `latest_result()` returns the
    # most recent archive in user_data/backtest_results/ whether or not THIS run
    # wrote it, so on failure it silently reports the PREVIOUS arm's numbers --
    # plausible, wrong, and indistinguishable from a real result. Observed
    # 2026-09-05: three windows of a broken arm all reported an identical
    # 59 trades / +3.86%, which was a stale file.
    result: dict
    if proc.returncode != 0:
        result = {"error": f"backtest FAILED (returncode {proc.returncode}); "
                           "no metrics recorded"}
        print(f"\n*** BACKTEST FAILED (rc={proc.returncode}) -- writing no metrics")
    else:
        latest = latest_result()
        if latest is not None and latest.stat().st_mtime < started:
            result = {"error": "backtest reported success but wrote no new result "
                               "file; refusing to report a stale one"}
            print("\n*** STALE RESULT FILE -- writing no metrics")
        else:
            result = summarize(latest)

    record = {
        "tag": args.tag,
        "window": label,
        "timerange": tr,
        "strategy": args.strategy,
        "prediction_threshold": args.threshold,
        # PROVENANCE, FIXED 2026-10-09 (feature 021 T035 gate review). The old field was
        # `"guards": args.guards`, i.e. the CLI FLAG, whose default is "on". It was recorded
        # on every arm regardless of what the strategy actually did, so all 90 trigger-only
        # arm records read `guards: on` while `SimpleStrategies/TriggerOnly.py` hardcodes
        # `entry_enable_guards: False` -- the effective setting was OFF and the record said
        # ON. Anyone auditing from the records alone would have concluded the arms tested
        # the trigger PLUS the guard chain, which is the opposite of what they tested.
        #
        # Two honest fields replace it, and neither claims more than it knows:
        "guards_requested": args.guards,
        #: the value this run WROTE into the params file, or None when it wrote none -- in
        #: which case the effective setting is the strategy class's own `buy_params`, which
        #: this script cannot see (it never imports the class; see exit_config_hash, which
        #: reads the params JSON for the same reason).
        "entry_enable_guards_written": guards_written(scratch, args.strategy),
        "overrides": overrides,
        "sell_overrides": sell_over,
        "trailing_overrides": trail_over,
        "exit_config_hash": exit_config_hash(scratch, args.strategy),
        # ALSO FIXED 2026-10-09: this was the literal `True`, an assertion dressed as a
        # measurement. Nothing in this script checks it, and it was cited as evidence in
        # feature 021 before anyone noticed it could not be false. The underlying claim is
        # structurally true but belongs in prose, not a boolean: the quote-volume headroom
        # check lives in `BaseStrategy.confirm_trade_entry` and runs unconditionally
        # outside plot/other runmodes, with `custom_stake_amount` CAPPING an oversized
        # stake rather than rejecting the trade.
        "volume_headroom_check": (
            "BaseStrategy.confirm_trade_entry, unconditional outside plot/other runmode; "
            "custom_stake_amount caps rather than rejects. NOT measured by this script."
        ),
        "returncode": proc.returncode,
        **result,
    }
    out = OUT_DIR / f"{args.tag}_{label}.json"
    out.write_text(json.dumps(record, indent=2))
    print(f"\nwrote {out}")
    print(json.dumps(record, indent=2))
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
