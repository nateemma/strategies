"""Training-budget diagnostic (Principle VI: diagnose before tuning).

WHY. The SC-011 probe incidentally produced two points on a learning curve, and the realised
policy was still moving sharply between them -- turnover 101 -> 21, exposure 0.54 -> 0.94
between 150k and 500k steps. Running the action-space screen at a non-converged budget would
turn it into a comparison of "which parameterisation converges faster by then", which is a
different question from "which is better".

It also separates two things the screen cannot: BUDGET variance from SEED variance. The 500k
probe's three seeds spanned 4.77 / 102.00 / 20.72 in turnover -- a 20x range. If seed variance
exceeds the action-space difference, the screen needs more seeds and no budget will rescue it.

SELECTION NOTE (SC-012). Choosing a budget from this is a selection on the screening fold. It
selects a BUDGET from TRAINING-RANGE behaviour only -- never a configuration, never a test
quantity -- and is pre-registered as such.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from research.labelfree import rl as RL                       # noqa: E402
from research.labelfree import stats as ST                    # noqa: E402
from research.labelfree.env import PooledTradingEnv           # noqa: E402
from research.labelfree.rl import episodes as EP              # noqa: E402
from research.labelfree.rl import panel as PN                 # noqa: E402
from research.labelfree.rl.rewards import SPECS               # noqa: E402
from research.labelfree.rl.train import PPO_PARAMS, make_env_fn   # noqa: E402

CHECKPOINTS = (150_000, 500_000, 1_000_000, 2_000_000, 3_000_000)


def rollout(model, panel, eps, continuous, band, seed, n_episodes=24):
    """Training-range behaviour only."""
    rng = np.random.default_rng(seed + 10_000)
    turns, exps, rews = [], [], []
    for _ in range(n_episodes):
        sl = None
        while sl is None:
            e = eps[int(rng.integers(len(eps)))]
            sl = PN.slice_episode(panel, e["pair"], e["start"], e["days"])
        env = PooledTradingEnv(*sl, reward_spec=SPECS["realised"](),
                               continuous=continuous, band=band)
        obs, _ = env.reset(seed=seed)
        tot, ws = 0.0, []
        while True:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(a)
            tot += r; ws.append(info["w"])
            if term or trunc:
                break
        turns.append(env.turnover); exps.append(float(np.mean(ws))); rews.append(tot)
    return (float(np.mean(turns)), float(np.mean(exps)), float(np.mean(rews)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fold", default=RL.SCREEN_FOLD)
    ap.add_argument("--action", choices=["discrete", "continuous"], default="discrete")
    ap.add_argument("--band", type=float, default=0.05)
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--out", default="/tmp/lf/rl_curve.json")
    a = ap.parse_args()

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    panel = PN.load_fold_panel(a.fold, train=True)
    eps = EP.sample(a.fold, n=2000, seed=0)
    assert not EP.validate(eps, a.fold)
    continuous = a.action == "continuous"
    seeds = [int(s) for s in a.seeds.split(",")]

    rows = []
    for seed in seeds:
        venv = DummyVecEnv([make_env_fn(panel, eps, "realised", continuous, a.band, seed)])
        model = PPO("MlpPolicy", venv, seed=seed, **PPO_PARAMS)
        done = 0
        for ck in CHECKPOINTS:
            t0 = time.time()
            model.learn(total_timesteps=ck - done, reset_num_timesteps=False,
                        progress_bar=False)
            done = ck
            tv, ex, rw = rollout(model, panel, eps, continuous, a.band, seed)
            rows.append({"seed": seed, "steps": ck, "turnover": tv,
                         "exposure": ex, "reward": rw, "seconds": round(time.time() - t0, 1)})
            print(f"  seed {seed} @ {ck//1000:>4}k: turnover={tv:8.2f} exposure={ex:.3f} "
                  f"reward={rw:+.4f} ({rows[-1]['seconds']}s)", flush=True)

    print(f"\n  {'steps':>8}{'turnover IQM':>15}{'exposure IQM':>15}{'reward IQM':>13}"
          f"{'turnover spread':>18}")
    summary = []
    for ck in CHECKPOINTS:
        g = [r for r in rows if r["steps"] == ck]
        tv = [r["turnover"] for r in g]
        s = {"steps": ck, "turnover_iqm": ST.iqm(tv),
             "exposure_iqm": ST.iqm([r["exposure"] for r in g]),
             "reward_iqm": ST.iqm([r["reward"] for r in g]),
             "turnover_min": min(tv), "turnover_max": max(tv)}
        summary.append(s)
        print(f"  {ck//1000:>7}k{s['turnover_iqm']:>15.2f}{s['exposure_iqm']:>15.3f}"
              f"{s['reward_iqm']:>13.4f}{f'{min(tv):.1f}-{max(tv):.1f}':>18}")

    # Has it flattened? Compare successive checkpoints on the reward, the quantity that matters.
    print(f"\n  step-to-step change in reward IQM:")
    for i in range(1, len(summary)):
        d = summary[i]["reward_iqm"] - summary[i - 1]["reward_iqm"]
        print(f"    {summary[i-1]['steps']//1000}k -> {summary[i]['steps']//1000}k: {d:+.4f}")

    Path(a.out).write_text(json.dumps({"fold": a.fold, "action": a.action,
                                       "rows": rows, "summary": summary}, indent=2))
    print(f"\n  -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
