# Quickstart: labelfree-rl-pooled-env

Validation scenarios. Repo root, sequential.

**Prerequisites**: repo `.venv`; gymnasium 1.3.0 / SB3 2.9.0 / sb3-contrib 2.9.0 / torch 2.13.0;
feathers 2021-01-01..2026-09-07; `pgrep -f 'freqtrade backtesting'` empty.

```bash
export PYTHONPATH=user_data/strategies
PY=.venv/bin/python ; RL=user_data/strategies/research/labelfree/rl
```

---

## S1 — The env reproduces the simulator bit-for-bit (R1)

```bash
$PY $RL/equivalence.py --folds f07 --n-paths 200
```

**Expected**: exit 0, `max|diff| = 0.00e+00` over all paths.

**This is the gate.** The env is a numpy restatement of an MLX simulator calibrated to the
arbiter at 0.00000000 bps. A restatement that is merely *close* is a different simulator. Cases
that must be covered: the `w_{-1}=0` opening boundary, a path still in position at the end, the
liquidity cap binding, and zero-volume bars.

---

## S2 — The environment can be reward-hacked (R5, SC-011)

```bash
$PY $RL/train.py --reward freqai_shaped --fold f07 --steps 200000 --seed 1 --probe
```

**Expected**: turnover measurably higher than a `realised` run at the same budget.

**If it is not, stop.** An environment whose reward cannot be *shown* to be hackable has not
been shown to be unhackable. This session has twice produced a control that passed vacuously —
feature 010's liveness control severed nothing, and feature 011's planted leak needed a
separation check before its threshold meant anything.

---

## S3 — No episode borrows from the test range (FR-009, R2)

```bash
$PY -c "
import sys; sys.path.insert(0,'user_data/strategies')
from research.labelfree.rl.episodes import sample, validate
eps = sample('f07', n=5000, seed=0)
bad = validate(eps, 'f07')
print('episodes:', len(eps), 'violations:', len(bad))"
```

**Expected**: `violations: 0`. The check includes the **two trailing bars** the reward needs —
an episode whose last decision borrows `open[i+1]`/`open[i+2]` from the test range leaks by
exactly two bars and nothing else would report it.

---

## S4 — Every swept parameter is live (SC-015)

```bash
$PY $RL/liveness.py
```

**Expected**: every parameter `LIVE`, each probed in its active regime.

**Watch `band_width`.** R4 predicts a *fixed* band avoids brief 02's bimodality, where a
*learned* band made turnover a coin flip between two attractors across seeds. If this probe
shows bimodality, the continuous arm inherits the defect and is reported, not tuned.

---

## S5 — Selection never touches the fold under evaluation (SC-012)

```bash
$PY -c "
import json,sys; sys.path.insert(0,'user_data/strategies')
recs=json.load(open('research/labelfree/rl/f07.json'))['selection_records']
bad=[r for r in recs if r['range']=='f07-test']
print('selection records:', len(recs), 'using the eval fold:', len(bad))"
```

**Expected**: `0`. Selecting on the test split is the specific defect the brief cites in
FreqAI's `best_model.zip`.

---

## S6 — The screen decides whether the 15-fold run is funded (SC-018)

```bash
$PY $RL/train.py --fold f07 --action discrete   --seeds 1,2,3,4,5 --steps 1500000
$PY $RL/train.py --fold f07 --action continuous --seeds 1,2,3,4,5 --steps 1500000
$PY $RL/evaluate.py --screen research/labelfree/rl/f07.json
```

**Expected**: `FUND` or `STOP`, against a deliberately weak bar — non-degenerate and
distinguishable from brief 1's policy. Marginal is funded; degenerate or identical is not.

---

## S7 — A policy that learned not to trade is inconclusive, not a return (R8, FR-017)

```bash
$PY -c "
import json
r=json.load(open('research/labelfree/rl/f07.json'))
for a in r['arms']:
    print(a['action_space'], 'mean_w=%.3f'%a['mean_exposure'],
          'degenerate=', a['degenerate'], 'return=', a.get('return','WITHHELD'))"
```

**Expected**: any arm below the participation floor shows `degenerate: True` and **no return**.
Zero exposure earns zero, which would otherwise rank above arms that lose money — brief 1's six
best-ranked configurations were all below the floor, two flat on 100% and 99.9% of bars.

---

## S8 — The head-to-head that decides the family (SC-009/SC-010)

```bash
$PY $RL/headtohead.py --rl research/labelfree/rl/pooled.json --brief1 <brief 1 policy>
```

**Expected**: paired IQM difference with interval, against the floor, both policies in the
simulator.

`MATCHED` **closes the family** — the differentiable estimator wins on lower variance and
backtestable state.

---

## S9 — Trial log clean, report written either way

```bash
$PY -c "
import sys; sys.path.insert(0,'user_data/strategies')
from research.labelfree.trials import audit
ok,i=audit(); print('PASS',ok,i['n_pre_registered'],i['problems'])"
```

**Expected**: `PASS True <n> []`. The protocol §6 report at
`research/label-free-study/briefs/results/04-<date>.md` is written **whether the feature is
accepted or killed** — the directional finding is the deliverable.
