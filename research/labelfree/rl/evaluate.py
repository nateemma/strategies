"""Screen gate and pooled evaluation (T035a, T037, T038, T042-T044).

--screen applies the SC-018 bar, pinned BEFORE the screen ran (analyze H1):
    non-degenerate  mean exposure above the participation floor
    distinguishable exposure paths differ from brief 1's on >5% of decision bars
It is deliberately weak -- time is not the binding constraint, so a marginal screen is
funded and only a degenerate or path-identical one is not.

--pooled trains one policy per fold on that fold's TRAINING range and evaluates it on that
fold's TEST range, then pools. No test quantity ever reaches a selection decision (SC-012).
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
from research.labelfree.admissibility import Arm              # noqa: E402
from research.labelfree.env import PooledTradingEnv           # noqa: E402
from research.labelfree.rl import episodes as EP              # noqa: E402
from research.labelfree.rl import panel as PN                 # noqa: E402
from research.labelfree.rl.rewards import SPECS               # noqa: E402


def evaluate_on_panel(model, panel, continuous: bool, band: float) -> dict:
    """Forward-only over every pair of a panel. Used for TEST ranges."""
    rew, exps, turns, bars = [], [], [], 0
    per_bar = []
    for pi, pair in enumerate(panel["pairs"]):
        X, P = panel["X"][pi], panel["P"][pi]
        C, V = panel["C"][pi], panel["V"][pi]
        n = len(P) - RL.EXEC_LAG_BARS
        if n <= 0:
            continue
        env = PooledTradingEnv(X[:n], P, C[:n], V[:n],
                               reward_spec=SPECS["realised"](),
                               continuous=continuous, band=band)
        obs, _ = env.reset(seed=0)
        tot, ws, step_rewards = 0.0, [], []
        while True:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(a)
            tot += r; ws.append(info["w"]); step_rewards.append(r)
            if term or trunc:
                break
        rew.append(tot); exps.append(float(np.mean(ws)))
        turns.append(env.turnover); bars += len(ws)
        per_bar.append(np.asarray(step_rewards, dtype=float))
    years = bars / (RL.BARS_PER_DAY * 365) if bars else 0.0
    trades = sum(turns) / 2.0                      # a round trip is two units of turnover
    # Mean across pairs, per bar: the TIME axis DSR and CSCV are defined over. Stored
    # compressed as 16 block returns (trials.block_returns), which is what closed the
    # SC-004 gap that four consecutive features reported as unevaluable.
    width = min((len(x) for x in per_bar), default=0)
    series = (np.mean([x[:width] for x in per_bar], axis=0).tolist() if width else [])
    return {"per_bar_returns": series,
            "reward": float(np.mean(rew)) if rew else 0.0,
            "mean_exposure": float(np.mean(exps)) if exps else 0.0,
            "turnover": float(np.mean(turns)) if turns else 0.0,
            "trades_per_year": float(trades / years) if years else 0.0,
            "n_pairs": len(rew)}


def screen(path: str, h2h_path: str) -> dict:
    d = json.loads(Path(path).read_text())
    ex = ST.iqm([r["train_mean_exposure"] for r in d["runs"]])
    non_degenerate = ex >= RL.MIN_EXPOSURE
    diff = json.loads(Path(h2h_path).read_text())["bars_differing"] \
        if Path(h2h_path).exists() else None
    distinguishable = diff is not None and diff > RL.DISTINGUISHABLE_MIN_BAR_FRACTION
    fund = non_degenerate and distinguishable
    return {"action": d["action"], "exposure_iqm": ex, "non_degenerate": non_degenerate,
            "bars_differing": diff, "distinguishable": distinguishable,
            "decision": "FUND" if fund else "STOP",
            "floor": RL.MIN_EXPOSURE,
            "distinguishability_bar": RL.DISTINGUISHABLE_MIN_BAR_FRACTION}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--screen")
    ap.add_argument("--h2h", default="research/labelfree/rl/headtohead.json")
    ap.add_argument("--pooled", action="store_true")
    ap.add_argument("--action", choices=["discrete", "continuous"], default="discrete")
    ap.add_argument("--band", type=float, default=0.05)
    ap.add_argument("--seeds", default="1,2,3,4,5")
    ap.add_argument("--steps", type=int, default=RL.STEPS_PER_FOLD)
    ap.add_argument("--folds", default="all")
    ap.add_argument("--out", default="research/labelfree/rl/pooled.json")
    a = ap.parse_args()

    if a.screen:
        s = screen(a.screen, a.h2h)
        print("SCREEN GATE (SC-018), bar pinned before the screen ran\n")
        print(f"  arm                 {s['action']}")
        print(f"  exposure IQM        {s['exposure_iqm']:.3f}  vs floor {s['floor']}"
              f"   -> {'non-degenerate' if s['non_degenerate'] else 'DEGENERATE'}")
        bd = f"{100*s['bars_differing']:.1f}%" if s['bars_differing'] is not None else "n/a"
        print(f"  bars differing      {bd}  vs bar "
              f">{100*s['distinguishability_bar']:.0f}%"
              f"   -> {'distinguishable' if s['distinguishable'] else 'NOT DISTINGUISHABLE'}")
        print(f"\n  DECISION: {s['decision']}")
        Path("research/labelfree/rl/screen_decision.json").write_text(json.dumps(s, indent=2))
        return 0

    if not a.pooled:
        raise SystemExit("pass --screen <file> or --pooled")

    decision = Path("research/labelfree/rl/screen_decision.json")
    if not decision.exists() or json.loads(decision.read_text())["decision"] != "FUND":
        raise SystemExit("--pooled refuses to run without a FUND screen on record (SC-018)")

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from research.labelfree.rl.train import PPO_PARAMS, make_env_fn

    folds = ([f.fold_id for f in EP.F.folds()] if a.folds == "all" else a.folds.split(","))
    seeds = [int(s) for s in a.seeds.split(",")]
    continuous = a.action == "continuous"
    run_id = trials.new_run_id()
    print(f"POOLED WALK-FORWARD — {len(folds)} folds x {len(seeds)} seeds, "
          f"{a.steps:,} steps each\n", flush=True)

    rows = []
    for fid in folds:
        tr_panel = PN.load_fold_panel(fid, train=True)
        te_panel = PN.load_fold_panel(fid, train=False)
        eps = EP.sample(fid, n=2000, seed=0)
        if EP.validate(eps, fid):
            raise SystemExit(f"{fid}: episode validation failed")
        for seed in seeds:
            tid = f"rl_pooled_{a.action}-s{seed}-{fid}@{run_id}"
            trials.pre_register({"trial_id": tid, "run_id": run_id, "feature": "012",
                                 "arm": f"rl_pooled_{a.action}", "seed": seed,
                                 "fold_id": fid, "steps": a.steps, **PPO_PARAMS})
            t0 = time.time()
            venv = DummyVecEnv([make_env_fn(tr_panel, eps, "realised", continuous,
                                            a.band, seed)])
            model = PPO("MlpPolicy", venv, seed=seed, **PPO_PARAMS)
            model.learn(total_timesteps=a.steps, progress_bar=False)
            ev = evaluate_on_panel(model, te_panel, continuous, a.band)   # TEST range
            series = ev.pop("per_bar_returns", None)
            ev.update(fold_id=fid, seed=seed, status="ok",
                      seconds=round(time.time() - t0, 1))
            trials.complete(tid, ev, per_bar_returns=series)
            rows.append(ev)
            print(f"  {fid} s{seed}: test reward={ev['reward']:+.4f} "
                  f"exposure={ev['mean_exposure']:.3f} "
                  f"trades/yr={ev['trades_per_year']:.1f} ({ev['seconds']}s)", flush=True)

    per_fold = []
    for fid in folds:
        g = [r for r in rows if r["fold_id"] == fid]
        per_fold.append({"fold_id": fid, "reward_iqm": ST.iqm([r["reward"] for r in g]),
                         "exposure_iqm": ST.iqm([r["mean_exposure"] for r in g]),
                         "positive_seeds": sum(1 for r in g if r["reward"] > 0),
                         "n_seeds": len(g)})
    pooled_reward = ST.iqm([r["reward"] for r in rows])
    pooled_exposure = ST.iqm([r["mean_exposure"] for r in rows])
    pooled_tpy = ST.iqm([r["trades_per_year"] for r in rows])
    folds_positive = sum(1 for f in per_fold if f["reward_iqm"] > 0)

    print(f"\n  {'fold':<7}{'reward IQM':>13}{'exposure':>11}{'seeds +ve':>12}")
    for f in per_fold:
        print(f"  {f['fold_id']:<7}{f['reward_iqm']:>13.4f}{f['exposure_iqm']:>11.3f}"
              f"{f'{f['positive_seeds']}/{f['n_seeds']}':>12}")
    print(f"\n  POOLED reward IQM {pooled_reward:+.4f}   exposure {pooled_exposure:.3f}   "
          f"trades/yr {pooled_tpy:.1f}")
    print(f"  folds positive: {folds_positive}/{len(folds)}  (SC-002 needs >=10/15)")

    arm = Arm(f"rl_{a.action}", pooled_reward, pooled_exposure, pooled_tpy)
    print(f"  admissibility: {'degenerate — ' + (arm.reason or '') if arm.degenerate else 'non-degenerate'}")

    Path(a.out).write_text(json.dumps(
        {"run_id": run_id, "action": a.action, "seeds": seeds, "steps": a.steps,
         "per_fold": per_fold, "rows": rows, "pooled_reward_iqm": pooled_reward,
         "pooled_exposure_iqm": pooled_exposure, "pooled_trades_per_year": pooled_tpy,
         "folds_positive": folds_positive, "n_folds": len(folds),
         "degenerate": arm.degenerate}, indent=2))
    print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
