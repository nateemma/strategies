# Quickstart — Validating Feature 003

Run from the repository root (`/Users/philprice95/projects/freqtrade`) with
`PYTHONPATH=user_data/strategies`. Execute steps **sequentially** — concurrent
jobs on this machine run slower than serialised ones.

## Prerequisites

- Price history present for the 11 traded pairs at 15m.
- Host model trained: `saved_data/NNNC_BreakoutOpen/` (sha `0d992dae`).
- Parked control arms present: `regime/arms/brkfix_W{1,2,3}.json`.

## Step 1 — Reproduce the screening figures (minutes, no training)

```
PYTHONPATH=user_data/strategies .venv/bin/python user_data/strategies/regime/breakout_tail.py
PYTHONPATH=user_data/strategies .venv/bin/python user_data/strategies/regime/tail_path.py
```

**Expect**: `H=48/0.005` at 9.07% of bars, median gain 2.83%, 26.0% clearing the
stop, reward:risk 0.57, time-to-peak 6.8h. If these differ materially, the data or
indicator set has changed and Phase 0 must be re-derived before proceeding.

## Step 2 — Verify the control is reproducible

```
PYTHONPATH=user_data/strategies .venv/bin/python user_data/strategies/regime/run_arm.py \
  --strategy NNNC_BreakoutOpen --window W1 --tag ctlcheck
```

**Expect**: 97 trades, −0.74%, `exit_config_hash` `bb3239ab7e52df1b`, and a
`trades_hash` equal to `brkfix_W1`'s. A mismatch means the host configuration
drifted and must be resolved first.

## Step 3 — Confirm the params-file pin does not change behaviour

After writing `NNNC/NNNC_BreakoutOpen.json`, re-run Step 2 with a new tag.

**Expect**: identical `trades_hash` to `brkfix_W1`. This is the R8 guarantee — the
file pins what is already running. **If the hash changes, stop**: the pin altered
behaviour and every historical comparison would be invalidated.

## Step 4 — Train and evaluate one candidate arm

```
cd <repo root> && PATH=.venv/bin:$PATH \
  zsh user_data/strategies/scripts/test_strat.sh -f -t 20220101-20250831 NNNC <ARM_CLASS>

for W in W1 W2 W3; do
  PYTHONPATH=user_data/strategies .venv/bin/python user_data/strategies/regime/run_arm.py \
    --strategy <ARM_CLASS> --window $W --tag <tag>
done
```

**Expect**: label counts in the training log matching the candidate's screened
density; a model hash distinct from the control's.

## Step 5 — Read the result correctly

Check in this order:

1. `exit_config_hash` matches the control ⇒ comparison is valid.
2. W1 `trades_hash` **differs** from the control ⇒ the change reached the trades.
   If identical, the verdict is **inconclusive_absorbed** and the returns are not
   evidence either way.
3. W1 return vs control, with trade counts. A delta below ~1pp on ~97 trades is
   **indistinguishable**, not an improvement.
4. W2/W3 are IN-SAMPLE. Report them; do not let them decide.

## Success looks like

- Every candidate has a screening record before any training (SC-001).
- Every arm is classified including the absorbed case (SC-003).
- The best arm's W1 return is stated against the control, in-sample windows
  separated (SC-004).
- No shared framework file left modified (SC-006).
- An explicit verdict on whether label mis-specification was material — including
  "it was not" (SC-008).

## Cleanup

Delete throwaway arm classes, their params files and their `saved_data/`
directories. Leave `NNNC/NNNC_BreakoutOpen.json` in place — it pins the control.
