"""PPO training over the pooled environment (T023, T025-T028, T031-T034).

Model selection uses TRAINING reward only; the test fold is untouched until the final pooled
evaluation. Every selection writes a SelectionRecord naming the quantity and the range it came
from, so the audit can VERIFY the claim rather than accept it (SC-012) -- selecting on the test
split is the specific defect brief 04 cites in FreqAI's `best_model.zip`.
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
from research.labelfree import trials                         # noqa: E402
from research.labelfree.env import PooledTradingEnv           # noqa: E402
from research.labelfree.rl import episodes as EP              # noqa: E402
from research.labelfree.rl import panel as PN                 # noqa: E402
from research.labelfree.rl.rewards import SPECS               # noqa: E402

PPO_PARAMS = {"n_steps": 512, "batch_size": 128, "learning_rate": 3e-4,
              "gamma": 0.99, "gae_lambda": 0.95, "ent_coef": 0.0, "verbose": 0}


def make_env_fn(panel, eps, reward_name, continuous, band, seed):
    """A fresh episode each reset, drawn from the pre-validated training-range list."""
    rng = np.random.default_rng(seed)

    def _f():
        import gymnasium as gym

        class _Cycler(gym.Wrapper):
            def __init__(self):
                sl = None
                while sl is None:
                    e = eps[int(rng.integers(len(eps)))]
                    sl = PN.slice_episode(panel, e["pair"], e["start"], e["days"])
                X, P, C, V = sl
                super().__init__(PooledTradingEnv(X, P, C, V,
                                                  reward_spec=SPECS[reward_name](),
                                                  continuous=continuous, band=band))

            def reset(self, **kw):
                sl = None
                while sl is None:
                    e = eps[int(rng.integers(len(eps)))]
                    sl = PN.slice_episode(panel, e["pair"], e["start"], e["days"])
                X, P, C, V = sl
                self.env = PooledTradingEnv(X, P, C, V,
                                            reward_spec=SPECS[reward_name](),
                                            continuous=continuous, band=band)
                return self.env.reset(**kw)
        return _Cycler()
    return _f


def train_one(panel, eps, *, reward_name: str, continuous: bool, band: float,
              seed: int, steps: int) -> dict:
    """One policy. Returns training-range diagnostics only — no test quantity is touched."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    venv = DummyVecEnv([make_env_fn(panel, eps, reward_name, continuous, band, seed)])
    model = PPO("MlpPolicy", venv, seed=seed, **PPO_PARAMS)
    t0 = time.time()
    model.learn(total_timesteps=steps, progress_bar=False)
    secs = round(time.time() - t0, 1)

    # Roll out on TRAINING episodes to measure behaviour. This is the selection quantity, and
    # it is recorded as such (SC-012).
    rng = np.random.default_rng(seed + 10_000)
    turns, exps, rews = [], [], []
    for _ in range(24):
        sl = None
        while sl is None:
            e = eps[int(rng.integers(len(eps)))]
            sl = PN.slice_episode(panel, e["pair"], e["start"], e["days"])
        env = PooledTradingEnv(*sl, reward_spec=SPECS[reward_name](),
                               continuous=continuous, band=band)
        obs, _ = env.reset(seed=seed)
        done = tot = 0.0, 
        tot, ws = 0.0, []
        while True:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(a)
            tot += r; ws.append(info["w"])
            if term or trunc:
                break
        turns.append(env.turnover); exps.append(float(np.mean(ws))); rews.append(tot)

    return {"model": model, "seconds": secs,
            "train_turnover": float(np.mean(turns)),
            "train_mean_exposure": float(np.mean(exps)),
            "train_reward": float(np.mean(rews)),
            "selection_record": {"decision": "final checkpoint",
                                 "criterion": "mean episode reward",
                                 "range": f"{eps[0]['fold_id']}-train"}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fold", default=RL.SCREEN_FOLD)
    ap.add_argument("--action", choices=["discrete", "continuous"], default="discrete")
    ap.add_argument("--band", type=float, default=0.05)
    ap.add_argument("--reward", choices=list(SPECS), default="realised")
    ap.add_argument("--seeds", default="1,2,3,4,5")
    ap.add_argument("--steps", type=int, default=RL.STEPS_PER_FOLD)
    ap.add_argument("--pairs", type=int, default=0, help="limit pairs (probe use only)")
    ap.add_argument("--out")
    ap.add_argument("--probe", action="store_true",
                    help="reward-hacking control; results can never be reported (T010)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if SPECS[a.reward]().probe and not a.probe:
        raise SystemExit(
            f"reward '{a.reward}' is a PROBE spec and exists to be hacked. It can never "
            "produce a reported figure; pass --probe to acknowledge that (T010).")

    fd = EP.fold(a.fold)
    pairs = list(fd.pair_universe)[:a.pairs] if a.pairs else list(fd.pair_universe)
    print(f"loading {a.fold} training panel ({len(pairs)} pairs)...", flush=True)
    panel = PN.load_fold_panel(a.fold, train=True, pairs=pairs)
    eps = EP.sample(a.fold, n=2000, seed=0, pairs=pairs)
    bad = EP.validate(eps, a.fold)
    if bad:
        raise SystemExit(f"episode validation failed: {len(bad)} violations, e.g. {bad[0]}")
    print(f"  {len(eps)} episodes, 0 violations\n", flush=True)

    run_id = trials.new_run_id()
    seeds = [int(s) for s in a.seeds.split(",")]
    results = []
    for seed in seeds:
        tid = f"rl_{a.action}_{a.reward}-s{seed}-{a.fold}@{run_id}"
        trials.pre_register({"trial_id": tid, "run_id": run_id, "feature": "012",
                             "arm": f"rl_{a.action}", "reward": a.reward, "seed": seed,
                             "fold_id": a.fold, "steps": a.steps, "band": a.band,
                             "probe": bool(a.probe), **PPO_PARAMS})
        if a.dry_run:
            continue
        r = train_one(panel, eps, reward_name=a.reward,
                      continuous=(a.action == "continuous"), band=a.band,
                      seed=seed, steps=a.steps)
        model = r.pop("model")
        r.update(status="ok", seed=seed, probe=bool(a.probe))
        trials.complete(tid, r)
        results.append(r)
        print(f"  seed {seed}: turnover={r['train_turnover']:.2f} "
              f"mean_w={r['train_mean_exposure']:.3f} reward={r['train_reward']:+.4f} "
              f"({r['seconds']}s)", flush=True)
        if a.out:
            model.save(str(Path(a.out).with_suffix("")) + f"_s{seed}")
    if a.dry_run:
        print("  dry run: configurations pre-registered, nothing trained")
        return 0

    out = {"run_id": run_id, "fold": a.fold, "action": a.action, "reward": a.reward,
           "probe": bool(a.probe), "seeds": seeds, "steps": a.steps,
           "turnover_iqm": ST.iqm([r["train_turnover"] for r in results]),
           "exposure_iqm": ST.iqm([r["train_mean_exposure"] for r in results]),
           "reward_iqm": ST.iqm([r["train_reward"] for r in results]),
           "runs": results}
    print(f"\n  IQM turnover {out['turnover_iqm']:.2f}  exposure {out['exposure_iqm']:.3f}  "
          f"reward {out['reward_iqm']:+.4f}")
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=2))
        print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
