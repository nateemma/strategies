# Contract: RL environment CLI

Run from the **repo root**, **sequentially**. Vectorised envs parallelise inside one run; two
runs never overlap.

```bash
export PYTHONPATH=user_data/strategies
PY=.venv/bin/python
RL=user_data/strategies/research/labelfree/rl
```

---

## 1. `equivalence.py` — THE CALIBRATION GATE

```
$PY $RL/equivalence.py --folds f07 --n-paths 200
```

**Contract**

- Generates random exposure paths, steps them through `research/labelfree/env.py`, and compares
  the accumulated rewards to `sim.net_returns` on the same path.
- **Requires bit-for-bit equality** (`rtol=0`), not closeness. A tolerance wide enough to absorb
  a boundary error would absorb the boundary error.
- Covers the cases a naive step gets wrong: the `w_{-1} = 0` opening boundary, a path that ends
  while still in position, the liquidity cap binding, and zero-volume bars.
- **Exits non-zero on any mismatch, and nothing downstream may run.** The env is a restatement
  of a simulator calibrated to the arbiter at 0.00000000 bps; an unverified restatement is a
  different simulator wearing its name.

---

## 2. `rewards.py --probe` — THE POSITIVE CONTROL

```
$PY $RL/train.py --reward freqai_shaped --fold f07 --steps 200000 --seed 1 --probe
```

**Contract**

- Trains briefly against the FreqAI-shaped reward (+25/entry, −1/idle bar, `pnl × 100` on exit).
- **Asserts turnover rises measurably** against a `realised`-reward run at the same budget.
- If it does not, the environment has not been shown to convey reward shape, and **SC-011 fails
  — no downstream number is admissible.**
- `--probe` output can never be reported as a result; it is a control.

---

## 3. `liveness.py` — every swept parameter proven live

```
$PY $RL/liveness.py
```

**Contract**

- One probe per swept parameter, each **in its active regime**, asserting the quantity it claims
  to control actually moves. `band_width` is the one to watch: R4 predicts a fixed band avoids
  brief 02's bimodality, and this is where that prediction is tested.
- Raises on a DEAD parameter. `train.py` refuses to run until this passes (SC-015).

---

## 4. `train.py` — one policy, one fold

```
$PY $RL/train.py --fold f07 --action discrete|continuous \
                 --seeds 1,2,3,4,5 --steps 1500000 --out research/labelfree/rl/f07.json
```

| flag | meaning |
|---|---|
| `--action` | `discrete` (MaskablePPO) or `continuous` (PPO through a fixed band) |
| `--steps` | PPO steps per seed |
| `--dry-run` | pre-register configurations, train nothing |

**Contract**

- Episodes sampled **only** from the fold's training range, window + 2 trailing bars inside it.
- Every configuration appended to `trials.jsonl` **before** it runs (FR-013).
- **Checkpoint selection uses training reward only**, and the quantity and its range are written
  to a `SelectionRecord` so the audit can verify rather than accept it (SC-012).
- Reports IQM over ≥5 seeds with a bootstrap interval, never a best seed. Seeds with identical
  trades count once.
- Every return carries mean exposure, trades/yr and mean hold time. A policy below the
  participation floor is emitted `degenerate: true` **with its return withheld** (FR-017).

---

## 5. `evaluate.py` — screen gate, then the pooled run

```
$PY $RL/evaluate.py --screen research/labelfree/rl/f07.json        # SC-018
$PY $RL/evaluate.py --pooled --folds all --out research/labelfree/rl/pooled.json
```

**Contract**

- `--screen` applies the pre-registered bar: the policy must be **non-degenerate** and
  **distinguishable from brief 1's** (per-seed distributions not identical). Deliberately weak —
  time is not the binding constraint, so a marginal screen is funded and only a degenerate or
  identical one is not.
- Prints `FUND` or `STOP`. `--pooled` refuses to run without a `FUND` screen on record.
- f07's contribution to the pooled verdict is reported **separately** (SC-019), because the
  screen selected on it.

---

## 6. `headtohead.py` — SC-009, the brief's actual question

```
$PY $RL/headtohead.py --rl research/labelfree/rl/pooled.json \
                      --brief1 research/labelfree/rl/brief1_policy.npz
```

**Contract**

- `--brief1` is a policy **re-trained and persisted by this feature** (T038a). Brief 1 saved no
  weights — `sweep.py`, `train.py` and `arms.py` write JSON results only — so the comparator is
  a fresh draw at this feature's seeds, **not** brief 1's reported numbers. The report must say
  so; quoting it as "brief 1's result" would be false.

- Both policies evaluated **in the simulator**, identical folds, costs and liquidity caps. Not
  through freqtrade — that is what keeps them comparable.
- Reports the paired IQM difference with its interval against the floor.
- Verdict: `RL BETTER BY THE FLOOR` · `MATCHED` · `WORSE`. **`MATCHED` closes the family** and
  records that the differentiable estimator is preferred for lower variance and backtestable
  state (SC-010).

---

## 7. `NNNC_RL.py` — §5 evaluation

**Written only if a configuration reaches the pooled evaluation.** Consults the policy in
`confirm_trade_entry` and `custom_exit`, which receive the candidate and the open `Trade`;
**not** in `populate_entry_trend`, which is vectorised and position-blind.

**Stated limitation, in the report rather than discovered in it**: these hooks can only veto
candidates the vectorised gate already proposed, so the policy cannot open a position the base
strategy never considered. The §5 figure is bounded by that, and the head-to-head is not.

---

## Exit codes

| code | meaning |
|---|---|
| 0 | success |
| 1 | a gate failed — equivalence, reward-hacking control, liveness, or selection audit |
| 2 | inputs missing |

A non-zero exit from `equivalence.py` or the reward probe **halts the feature**.
