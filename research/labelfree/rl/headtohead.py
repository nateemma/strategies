"""SC-009: does SAMPLING beat SMOOTHING on the same environment? (T038a-T041)

This is brief 04's actual question. It is NOT "is RL deployable" -- that is SC-001, a separate
bar -- and the two can have different answers.

THE COMPARATOR IS RE-TRAINED, NOT RECOVERED (analyze C1). Brief 1 persists no trained policy:
sweep.py, train.py and arms.py write JSON results only, never weights. So brief 1's best
configuration is trained here, on this fold, at these seeds. Its numbers are a FRESH DRAW and
must never be quoted as brief 1's reported figures.

BOTH POLICIES RUN IN THE SIMULATOR, never through freqtrade. That is what keeps them
comparable: identical folds, identical costs, identical liquidity caps, one evaluation path.
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

# Brief 1's best configuration on its screening fold: utility objective, PENALTY turnover
# control, gamma 0.5 -- reported at -0.07060 with mean_w 0.130. That figure is NOT reused;
# only the configuration is.
# EPOCHS IS PART OF THE CONFIGURATION AND MUST MATCH WHAT BRIEF 1 RAN.
# TrainConfig's dataclass default is 40, but brief 1 never used it: sweep.py's CLI default
# is 4 and feature 010's arms.py used 2. Taking the dataclass default meant comparing
# against a configuration brief 1 never trained -- and doing 10x the work to do it. Caught
# 2026-09-13 after a 48-minute run produced no completed seed.
#
# This is the inverse of the defect this study keeps meeting: usually the CONFIGURED value
# silently fails to apply; here the DEFAULT silently differed from what was applied.
BRIEF1_CFG = {"objective": "utility", "turnover_control": "penalty", "gamma": 0.5,
              "epochs": 4}


def train_brief1(fold_id: str, seed: int):
    """Re-train brief 1's best configuration. Returns (policy, seconds)."""
    from research.labelfree.train import TrainConfig, train_fold
    panel = PN.load_fold_panel(fold_id, train=True)
    cfg = TrainConfig(seed=seed, **BRIEF1_CFG)
    t0 = time.time()
    pol, _hist = train_fold(panel["X"], panel["P"], panel["C"], panel["V"], cfg,
                            mask=panel["M"])
    return pol, round(time.time() - t0, 1)


def brief1_exposure(pol, panel, pair_idx: int) -> np.ndarray:
    """Brief 1's exposure path on one pair, forward-only."""
    import mlx.core as mx
    x = mx.array(panel["X"][pair_idx][None, ...].astype(np.float32))
    w = pol(x)
    mx.eval(w)
    return np.asarray(w).ravel()


def replay_exposure(w: np.ndarray, X, P, C, V) -> dict:
    """Score ANY exposure path through the env, so both policies are measured identically."""
    env = PooledTradingEnv(X, P, C, V, reward_spec=SPECS["realised"]())
    env.reset(seed=0)
    tot, ws = 0.0, []
    for i in range(env.n):
        wi = float(np.clip(w[i] if i < len(w) else 0.0, 0.0, 1.0))
        env.w = env.w            # explicit: the env applies the target directly
        _, r, term, trunc, info = env.step(np.array([wi], dtype=np.float32)) \
            if env.continuous else env.step(1 if wi >= 0.5 else 0)
        tot += r; ws.append(info["w"])
        if term or trunc:
            break
    return {"reward": tot, "mean_exposure": float(np.mean(ws)) if ws else 0.0,
            "turnover": env.turnover, "path": np.array(ws)}


def rl_exposure(model, X, P, C, V) -> dict:
    env = PooledTradingEnv(X, P, C, V, reward_spec=SPECS["realised"]())
    obs, _ = env.reset(seed=0)
    tot, ws = 0.0, []
    while True:
        a, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(a)
        tot += r; ws.append(info["w"])
        if term or trunc:
            break
    return {"reward": tot, "mean_exposure": float(np.mean(ws)),
            "turnover": env.turnover, "path": np.array(ws)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fold", default=RL.SCREEN_FOLD)
    ap.add_argument("--seeds", default="1,2,3,4,5")
    ap.add_argument("--rl-models", default="research/labelfree/rl/screen_discrete")
    ap.add_argument("--out", default="research/labelfree/rl/headtohead.json")
    a = ap.parse_args()

    from stable_baselines3 import PPO

    seeds = [int(s) for s in a.seeds.split(",")]
    panel = PN.load_fold_panel(a.fold, train=True)
    eps = EP.sample(a.fold, n=400, seed=0)
    run_id = trials.new_run_id()

    print(f"HEAD-TO-HEAD — {a.fold}, {len(seeds)} seeds, both policies in the simulator",
          flush=True)
    print(f"  comparator: brief 1 {BRIEF1_CFG} — RE-TRAINED here, not brief 1's figures\n",
          flush=True)

    rows = []
    for seed in seeds:
        tid = f"brief1_comparator-s{seed}-{a.fold}@{run_id}"
        trials.pre_register({"trial_id": tid, "run_id": run_id, "feature": "012",
                             "arm": "brief1_comparator", "seed": seed, "fold_id": a.fold,
                             "retrained": True, **BRIEF1_CFG})
        b1, secs = train_brief1(a.fold, seed)
        model = PPO.load(f"{a.rl_models}_s{seed}")

        per_pair = []
        for pi, pair in enumerate(panel["pairs"]):
            sl = PN.slice_episode(panel, pair, eps[0]["start"], 60)
            if sl is None:
                continue
            X, P, C, V = sl
            w_b1 = brief1_exposure(b1, panel, pi)[:len(X)]
            r_b1 = replay_exposure(w_b1, X, P, C, V)
            r_rl = rl_exposure(model, X, P, C, V)
            n = min(len(r_b1["path"]), len(r_rl["path"]))
            differ = float(np.mean(np.abs(r_b1["path"][:n] - r_rl["path"][:n]) > 1e-6))
            per_pair.append({"pair": pair, "brief1": r_b1["reward"], "rl": r_rl["reward"],
                             "brief1_w": r_b1["mean_exposure"], "rl_w": r_rl["mean_exposure"],
                             "bars_differing": differ})
        row = {"seed": seed,
               "brief1": float(np.mean([p["brief1"] for p in per_pair])),
               "rl": float(np.mean([p["rl"] for p in per_pair])),
               "brief1_w": float(np.mean([p["brief1_w"] for p in per_pair])),
               "rl_w": float(np.mean([p["rl_w"] for p in per_pair])),
               "bars_differing": float(np.mean([p["bars_differing"] for p in per_pair])),
               "brief1_train_seconds": secs}
        row["delta"] = row["rl"] - row["brief1"]
        trials.complete(tid, {k: v for k, v in row.items()})
        rows.append(row)
        print(f"  seed {seed}: brief1={row['brief1']:+.4f} rl={row['rl']:+.4f} "
              f"delta={row['delta']:+.4f}  bars differing {100*row['bars_differing']:.1f}% "
              f"(b1 train {secs}s)", flush=True)

    d = [r["delta"] for r in rows]
    lo, hi = ST.bootstrap_ci(d)
    iqm = ST.iqm(d)
    diff_frac = float(np.mean([r["bars_differing"] for r in rows]))
    print(f"\n  paired delta (RL - brief1)  IQM {iqm:+.4f}  mean {np.mean(d):+.4f}")
    print(f"  bootstrap 95% CI [{lo:+.4f}, {hi:+.4f}]   RL better in {sum(x>0 for x in d)}/{len(d)} seeds")
    print(f"  exposure paths differ on {100*diff_frac:.1f}% of bars "
          f"(SC-018 bar: >{100*RL.DISTINGUISHABLE_MIN_BAR_FRACTION:.0f}%)")

    # THE FLOOR APPLIES TO BOTH SIDES (FR-017). A comparator below the participation floor
    # is INCONCLUSIVE and its return is withheld, so no delta against it is admissible --
    # a policy that abstains is not a policy that performs. Applying the floor only to the
    # arm under test is how brief 02 came to measure arms against a P&L control that lost
    # 52% in a rising market: the comparator drove the verdict, not the arm.
    b1_w = float(np.mean([r["brief1_w"] for r in rows]))
    rl_w = float(np.mean([r["rl_w"] for r in rows]))
    b1_degenerate = b1_w < RL.MIN_EXPOSURE
    rl_degenerate = rl_w < RL.MIN_EXPOSURE
    print(f"\n  mean exposure — brief1 {b1_w:.4f}  RL {rl_w:.3f}  "
          f"floor {RL.MIN_EXPOSURE}")

    resolved = lo > 0 or hi < 0
    if b1_degenerate or rl_degenerate:
        who = "comparator" if b1_degenerate else "RL policy"
        verdict = "INCONCLUSIVE — DEGENERATE COMPARATOR" if b1_degenerate \
            else "INCONCLUSIVE — DEGENERATE ARM"
        print(f"  *** the {who} is below the participation floor: its return is WITHHELD "
              f"(FR-017) ***")
        print(f"  No delta against it is admissible. The raw delta would have read "
              f"{'WORSE' if resolved and iqm < 0 else 'MATCHED'}.")
    else:
        verdict = ("RL BETTER BY THE FLOOR" if resolved and iqm > 0 else
                   "WORSE" if resolved and iqm < 0 else "MATCHED")
    print(f"\n  SC-009 VERDICT: {verdict}")
    if verdict == "MATCHED":
        print("  SC-010: matched closes the family — the differentiable estimator is")
        print("  preferred for lower variance and backtestable state.")
    Path(a.out).write_text(json.dumps(
        {"run_id": run_id, "fold": a.fold, "comparator": BRIEF1_CFG, "retrained": True,
         "rows": rows, "delta_iqm": iqm, "delta_ci": [lo, hi], "verdict": verdict,
         "bars_differing": diff_frac,
         "brief1_mean_exposure": b1_w, "rl_mean_exposure": rl_w,
         "brief1_degenerate": b1_degenerate, "rl_degenerate": rl_degenerate,
         "participation_floor": RL.MIN_EXPOSURE}, indent=2))
    print(f"  -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
