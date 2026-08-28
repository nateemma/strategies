#!/usr/bin/env python3
"""
WGAN class-mean-fix A/B runner (docs/GAN_TODO.md #5 powered protocol).

Three arms, paired seeds, at the POWERED operating point (prediction_threshold
0.45, ~580 trades) and a PINNED timerange:

    nogan  non-GAN control            (NNNC_MLX)
    w0     WGAN, class-mean term OFF  (NNNC_WGAN_MLX + pre-fix GAN artifact)
    w2     WGAN, class-mean term ON   (NNNC_WGAN_MLX + fixed GAN artifact)

w0 and w2 use the SAME strategy code and differ only in which GAN generator is
installed at saved_data/GANs_PostScale/wgan/ — GAN artifacts are keyed by
GANType, not by builder name, so the arms are run in sequence:

    train GAN (CreateWGAN_NoCond) -> run w0 -> train GAN (CreateWGAN) -> run w2

Each (arm, seed) gets its own throwaway strategy class so classifier weights
never collide and PRODUCTION weights are never touched or deleted (no
get_model_path override, so each temp class trains under its own name).

Usage (from repo root):
    python user_data/strategies/scripts/wgan_ab.py w2 --seeds 1 7 13 42
    python user_data/strategies/scripts/wgan_ab.py nogan --seeds 1 7 13 42
    # optional: --timerange 20240629-20260619 (default; pinned for comparability)
"""

import argparse
import os
import pathlib
import re
import subprocess
import sys


REPO = pathlib.Path(__file__).resolve().parents[3]
STRAT_DIR = REPO / "user_data" / "strategies" / "NNNC"
LOGDIR = REPO / "user_data" / "strategies" / "scripts" / "sweep_logs"
TEST_STRAT = REPO / "user_data" / "strategies" / "scripts" / "test_strat.sh"

# The A/B window used by every prior powered run in GAN_TODO #5. Pinned: -n/-o
# slide one day per calendar day and runs taken on different days would not be
# comparable.
DEFAULT_TIMERANGE = "20240629-20260619"

# Loosened entry gate — the powered operating point. Same value as
# NNNC_MLX_P0 / NNNC_DDPM_MLX_P0.
PRED_THRESHOLD = 0.45

ARMS = {
    "nogan": "NNNC_MLX",
    "w0": "NNNC_WGAN_MLX",
    "w2": "NNNC_WGAN_MLX",
}


def make_temp_strategy(arm: str, seed: int) -> tuple[str, pathlib.Path]:
    base = ARMS[arm]
    cls = f"WGANAB_{arm}_s{seed}"
    content = (
        "import sys\n"
        "from pathlib import Path\n"
        "sys.path.append(str(Path(__file__).parent))\n"
        f"from {base} import {base}\n"
        "from TrainSeedStrategyMixin import TrainSeedStrategyMixin\n\n\n"
        f"class {cls}(TrainSeedStrategyMixin, {base}):\n"
        f"    train_seed = {seed}\n"
        f"    buy_params = {{**{base}.buy_params, "
        f"'prediction_threshold': {PRED_THRESHOLD}}}\n"
    )
    path = STRAT_DIR / f"{cls}.py"
    path.write_text(content)
    return cls, path


def run_backtest(cls: str, timerange: str) -> str:
    cmd = ["zsh", str(TEST_STRAT), "-f", "-t", timerange, "NNNC", cls]
    env = {**os.environ}
    env["PATH"] = str(REPO / ".venv" / "bin") + ":" + env.get("PATH", "")
    proc = subprocess.run(
        cmd, cwd=str(REPO), env=env, capture_output=True, text=True
    )
    return proc.stdout + "\n" + proc.stderr


def parse(out: str) -> dict:
    def g(pat, default="?"):
        m = re.search(pat, out)
        return m.group(1) if m else default

    return {
        "profit": g(r"Total profit %\s*[│|]\s*(-?[0-9.]+)%"),
        "calmar": g(r"Calmar \(closed trades\)\s*[│|]\s*(-?[0-9.]+)"),
        "dd": g(r"Absolute drawdown\s*[│|]\s*[0-9.]+ USDT \((-?[0-9.]+)%\)"),
        "pf": g(r"Profit factor\s*[│|]\s*(-?[0-9.]+)"),
        "trades": g(r"Total/Daily Avg Trades\s*[│|]\s*([0-9]+)"),
        "sigma": g(r"sigma_syn/sigma_real[^0-9-]*(-?[0-9.]+)"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", choices=ARMS.keys())
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 7, 13, 42])
    ap.add_argument("--timerange", default=DEFAULT_TIMERANGE)
    ap.add_argument(
        "--keep",
        action="store_true",
        help="Keep the generated temp strategy files (default: remove).",
    )
    args = ap.parse_args()

    LOGDIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in args.seeds:
        cls, path = make_temp_strategy(args.arm, seed)
        print(f"[wgan_ab] arm={args.arm} seed={seed} -> {cls} ...", flush=True)
        try:
            out = run_backtest(cls, args.timerange)
            (LOGDIR / f"{cls}.log").write_text(out)
            m = parse(out)
        finally:
            if not args.keep:
                path.unlink(missing_ok=True)
                for f in (path.parent / "__pycache__").glob(f"{cls}.*.pyc"):
                    f.unlink(missing_ok=True)
        rows.append((seed, m))
        print(
            f"    profit={m['profit']}%  calmar={m['calmar']}  dd={m['dd']}%  "
            f"pf={m['pf']}  trades={m['trades']}",
            flush=True,
        )

    print(f"\n=== WGAN A/B arm={args.arm} ({args.timerange}, "
          f"pred_thr={PRED_THRESHOLD}) ===")
    print(f"{'seed':>5} | {'profit%':>8} | {'calmar':>7} | {'dd%':>5} | "
          f"{'pf':>5} | {'trades':>6}")
    print("-" * 52)
    for seed, m in rows:
        print(f"{seed:>5} | {m['profit']:>8} | {m['calmar']:>7} | {m['dd']:>5} | "
              f"{m['pf']:>5} | {m['trades']:>6}")
    vals = [float(m["profit"]) for _, m in rows if m["profit"] != "?"]
    if vals:
        mean = sum(vals) / len(vals)
        print(f"\nmean profit% = {mean:.2f}  (n={len(vals)})")
    print("(per-run logs in scripts/sweep_logs/)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
