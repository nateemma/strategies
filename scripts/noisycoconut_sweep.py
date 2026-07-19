#!/usr/bin/env python3
"""
NoisyCoconut sigma-sweep runner.

For each sigma, generates a throwaway subclass of the chosen NoisyCoconut
strategy (so no committed strategy file is edited and the production weights are
reused via the inherited get_model_path), runs a backtest via test_strat.sh,
parses the summary metrics, and prints a single comparison table. Temp files are
removed afterwards.

Usage (from repo root):
    python user_data/strategies/scripts/noisycoconut_sweep.py latent 0.0 0.05 0.2 0.4
    python user_data/strategies/scripts/noisycoconut_sweep.py input 0.0 0.01 0.02 0.05 0.1
    # optional: --ndays 720 --offset 30   (defaults shown)

sigma=0.0 reproduces production exactly — a built-in sanity/anchor row.
"""

import argparse
import pathlib
import re
import subprocess
import sys


REPO = pathlib.Path(__file__).resolve().parents[3]
STRAT_DIR = REPO / "user_data" / "strategies" / "NNNC"
LOGDIR = REPO / "user_data" / "strategies" / "scripts" / "sweep_logs"
TEST_STRAT = REPO / "user_data" / "strategies" / "scripts" / "test_strat.sh"

FAMILY = {
    "latent": "NNNC_DDPM_MLX_Noisy",
    "input": "NNNC_DDPM_MLX_InJit",
}


def sigma_tag(sigma: float) -> str:
    return f"s{str(sigma).replace('.', '_')}"


def make_temp_strategy(base: str, sigma: float) -> tuple[str, pathlib.Path]:
    cls = f"NNNCSweep_{base}_{sigma_tag(sigma)}"
    content = (
        "import sys\n"
        "from pathlib import Path\n"
        "sys.path.append(str(Path(__file__).parent))\n"
        f"from {base} import {base}\n\n\n"
        f"class {cls}({base}):\n"
        f"    noisy_sigma = {sigma}\n"
    )
    path = STRAT_DIR / f"{cls}.py"
    path.write_text(content)
    return cls, path


def run_backtest(cls: str, ndays: int, offset: int) -> str:
    cmd = ["zsh", str(TEST_STRAT), "-n", str(ndays), "-o", str(offset), "NNNC", cls]
    env = {**__import__("os").environ}
    env["PATH"] = str(REPO / ".venv" / "bin") + ":" + env.get("PATH", "")
    proc = subprocess.run(
        cmd, cwd=str(REPO), env=env, capture_output=True, text=True
    )
    return proc.stdout + "\n" + proc.stderr


def parse(out: str) -> dict:
    def g(pat, default="?"):
        m = re.search(pat, out)
        return m.group(1) if m else default

    stops = g(r"[│|]\s*stop_loss\s*[│|]\s*([0-9]+)\s*[│|]")
    return {
        "profit": g(r"Total profit %\s*[│|]\s*(-?[0-9.]+)%"),
        "calmar": g(r"Calmar \(closed trades\)\s*[│|]\s*(-?[0-9.]+)"),
        "dd": g(r"Absolute drawdown\s*[│|]\s*[0-9.]+ USDT \((-?[0-9.]+)%\)"),
        "pf": g(r"Profit factor\s*[│|]\s*(-?[0-9.]+)"),
        "trades": g(r"Total/Daily Avg Trades\s*[│|]\s*([0-9]+)"),
        "stops": stops,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("family", choices=FAMILY.keys())
    ap.add_argument("sigmas", nargs="+", type=float)
    ap.add_argument("--ndays", type=int, default=720)
    ap.add_argument("--offset", type=int, default=30)
    args = ap.parse_args()

    base = FAMILY[args.family]
    LOGDIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for sigma in args.sigmas:
        cls, path = make_temp_strategy(base, sigma)
        print(f"[sweep] {base} sigma={sigma} -> {cls} ...", flush=True)
        try:
            out = run_backtest(cls, args.ndays, args.offset)
            (LOGDIR / f"{cls}.log").write_text(out)
            m = parse(out)
        finally:
            path.unlink(missing_ok=True)
            pyc = path.parent / "__pycache__"
            for f in pyc.glob(f"{cls}.*.pyc"):
                f.unlink(missing_ok=True)
        rows.append((sigma, m))
        print(
            f"    profit={m['profit']}%  calmar={m['calmar']}  dd={m['dd']}%  "
            f"pf={m['pf']}  trades={m['trades']}  stops={m['stops']}",
            flush=True,
        )

    print(f"\n=== NoisyCoconut sweep: {base} ({args.ndays}d/-{args.offset}) ===")
    print(f"{'sigma':>7} | {'profit%':>8} | {'calmar':>7} | {'dd%':>5} | "
          f"{'pf':>5} | {'trades':>6} | {'stops':>5}")
    print("-" * 60)
    for sigma, m in rows:
        print(f"{sigma:>7} | {m['profit']:>8} | {m['calmar']:>7} | {m['dd']:>5} | "
              f"{m['pf']:>5} | {m['trades']:>6} | {m['stops']:>5}")
    print("\n(sigma=0.0 should match production; per-run logs in scripts/sweep_logs/)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
