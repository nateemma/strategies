# Implementation Plan: labelfree-rl-pooled-env

**Branch**: `012-labelfree-rl-pooled-env` | **Date**: 2026-09-12 | **Spec**: [spec.md](spec.md)

## Summary

Build a pooled trading environment whose reward is the realised economics of brief 1's
simulator, train one shared policy across all pairs with PPO, and answer one question: **does
sampling path-dependent decisions beat smoothing them?** The head-to-head against brief 1's
differentiable policy is the brief's actual test; protocol §5 is a separate acceptance bar.

Compute is staged. One screening fold decides whether the full 15-fold run is funded, against a
deliberately weak pre-registered bar — non-degenerate and distinguishable from brief 1's policy.

## Technical Context

**Language/Version**: Python 3.12.13 (repo `.venv`)
**Primary Dependencies**: gymnasium 1.3.0, stable-baselines3 2.9.0, sb3-contrib 2.9.0
(`MaskablePPO`), torch 2.13.0 — all installed, no new dependency. MLX only for the equivalence
test and brief 1's policy in the head-to-head.
**Storage**: JSON artefacts under `research/labelfree/rl/`; trial log is the existing
`research/labelfree/trials.jsonl`
**Testing**: pytest under `research/labelfree/tests/`
**Target Platform**: macOS / Apple Silicon, **CPU** (SB3 has no MPS path worth using for an MLP)
**Project Type**: research module inside the strategies repository
**Performance Goals**: ~1–2M PPO steps per fold; vectorised envs across the 10 performance
cores; runs strictly sequential
**Constraints**: the env step must reproduce `sim.net_returns` bit-for-bit; no selection may
consume a test-fold quantity; the reward carries no entry bonus, idle penalty or take-profit
multiplier; all runs sequential
**Scale/Scope**: 10 pairs, 15 folds, ≥5 seeds, 2 action parameterisations

**Measured inputs**:

| quantity | value | source |
|---|---|---|
| pooled archive | 378 trades / 15 folds | `regime/wf/pooled_nnnc.json` |
| screening fold | **f07** | brief 04 experiment 2 |
| brief 1 comparator (f08) | `utility-penalty-g0.5`, −0.0706, mean_w 0.130 | brief 01 result |
| liquidity cap | 10% of candle quote volume | `sim.LIQUIDITY_FRACTION` |
| cores | 14 physical / 10 performance | measured |

## Constitution Check

| Principle | Status | How |
|---|---|---|
| I. Deployment regime | **PASS** | Pooled evaluation over the harness's 15 anchored folds |
| II. Persistence, distant eras (NON-NEGOTIABLE) | **PASS** | SC-002 requires W1/W2/W3 single-window; W3 is 2023, ~2.5 yr from W1 |
| III. Learnability is not edge | **PASS** | Training reward is never the verdict; SC-001 is P&L, SC-009 is a P&L head-to-head |
| IV. Parameters actually live | **PASS** | FR-019/SC-015; band width in particular (R4) |
| V. Lookahead-clean is not realistic | **PASS** | Reward carries fee, spread, slippage and the liquidity cap; R2 pins the execution lag |
| VI. Diagnose before tuning | **PASS** | The reward-hacking probe (R5) runs before any tuning, and the equivalence test (R1) before any training |
| VII. Validate cheaply first | **PASS** | Staged budget: screening fold gates the 15-fold run (SC-018) |
| VIII. One parameter space at a time | **PASS** | Action parameterisation screened first and locked; seeds thereafter. No algorithm comparison (FR-011) |
| IX. Orthogonality | **PASS** | Compared against brief 1's policy — the thing it would replace |
| X. No per-pair overrides | **PASS** | One shared policy, no per-pair parameters (FR-008) |

**Repository constraints**: commits only to `user_data/strategies`; file-by-file staging;
`Framework/` untouched; no parameter dictionary redefined wholesale.

**Experiment gates**: pinned ranges (fold plan), distant-era window (W3), realistic execution
(cap + costs in reward), liveness proof (FR-019), ≥5 seeds with IQM (FR-014), trade count
reported (FR-016), scalers N/A (no change to the primary's feature list).

## Project Structure

```
specs/012-labelfree-rl-pooled-env/
├── spec.md · plan.md · research.md · data-model.md · quickstart.md
├── contracts/rl-cli.md
└── checklists/requirements.md

research/labelfree/
├── env.py            # gymnasium env; numpy step economics (R1)
└── rl/
    ├── __init__.py   # pinned constants
    ├── episodes.py   # (pair, start, length) sampling from the TRAINING range only
    ├── equivalence.py# step path == sim.net_returns, bit-for-bit (R1) — the calibration gate
    ├── rewards.py    # the specified reward + the FreqAI-shaped probe reward (R5)
    ├── train.py      # SB3 PPO / MaskablePPO; selection on training reward only
    ├── evaluate.py   # policy -> exposure path -> pooled evaluation
    ├── headtohead.py # SC-009: RL vs brief 1's policy, identical env and folds
    └── liveness.py   # probes for every swept parameter (SC-015)

NNNC/
└── NNNC_RL.py        # WRITTEN ONLY IF a configuration reaches the §5 evaluation (R3)
```

**Structure decision**: `env.py` sits where the starter prompt names it; everything else lives
under `research/labelfree/rl/` and reuses `trials.py`, `stats.py`, `folds.py` and `costs.py`
rather than duplicating them. `sim.py` is **read, never modified** — it is calibrated to the
arbiter at 0.00000000 bps and is a shared dependency of briefs 2, 5 and 6.

## Phases

- **Phase 0 — Research** *(complete)*: `research.md`, 8 decisions.
- **Phase 1 — Design** *(complete)*: `data-model.md`, `contracts/rl-cli.md`, `quickstart.md`.
- **Phase 2 — Tasks**: `/speckit-tasks`. First tasks are the equivalence gate and the
  reward-hacking probe; last are the §6 report and the trial-log audit.
- **Phase 3 — Implement**: after a clean `/speckit-analyze`. Sequential runs only.

## Complexity Tracking

| Deviation | Why needed | Simpler alternative rejected because |
|---|---|---|
| **Env step reimplemented in numpy** rather than calling `sim.net_returns` | MLX per-scalar dispatch inside a 10^8-step CPU rollout loop is the wrong tool; SB3 workers are separate processes | Calling `net_returns` on a 2-bar window each step (rejected: the `w[-1]=0` boundary would re-zero the carried position every step — a silent, plausible-looking bug); refactoring `sim.py` (rejected: shared dependency of briefs 2/5/6, calibrated to the arbiter). **Mitigation: bit-for-bit equivalence test as a blocking gate.** |
| **Evaluation through per-candidate hooks** rather than vectorised entry population | A position-aware policy cannot be expressed as a position-blind boolean column | Dropping position state (rejected: it is the brief's stated advantage over FreqAI). **Cost, stated in the report: the policy can only veto candidates the vectorised gate already proposed, so it cannot open a position the base strategy never considered.** |
| **Fixed no-trade band** rather than brief 02's learned band | A learned band produced two attractors and made turnover a coin flip across seeds | Reusing `ClassifierEncoderPolicy`'s band head (rejected: inherits the bimodality). **This is a prediction; SC-015's band-width probe tests it, and if the arm turns out bimodal it is reported, not tuned.** |
| **Screening fold counted in the pooled verdict** | Excluding f07 costs a fold from an already thin 15 | Excluding it (rejected on power). **Mitigation: the screen selects only between two action parameterisations, and f07's contribution is reported separately (SC-019).** |
