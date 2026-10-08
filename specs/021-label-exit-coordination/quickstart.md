# Quickstart: Label-Exit Coordination

Runnable validation. Every command assumes:

```bash
cd /Users/philprice95/projects/freqtrade
source .venv/bin/activate
export PYTHONPATH="user_data/strategies:user_data/strategies/NNNC:user_data/strategies/utils"
```

A live dry-run (`MomentumRegimeBasket15mFast`) is running. It does not use the NN exit path,
but step 1 modifies `Framework/BaseStrategy.py`, which it does inherit — so step 1's
verification is not optional.

## 1. The exit policy is in one place, and live behaviour is unchanged

```bash
# Generate the characterisation fixture from the CURRENT code, BEFORE refactoring.
python -m pytest user_data/strategies/Framework/tests/test_exit_policy_single_source.py \
  -k characterisation --generate-fixture

# ... refactor ...

python -m pytest user_data/strategies/Framework/tests/test_exit_policy_single_source.py -v
```

**Expect**: the characterisation test passes against the pre-refactor fixture (SC-001), and
the allowlist test passes (SC-002).

**Prove the duplication test actually bites** — a test that cannot fail is worthless:

```bash
printf 'atr_stoploss_multiplier = 2.5\nstop = -atr_stoploss_multiplier * 0.01\n' \
  > user_data/strategies/utils/_dup_probe.py
python -m pytest user_data/strategies/Framework/tests/test_exit_policy_single_source.py \
  -k duplication    # EXPECT FAILURE naming _dup_probe.py
rm user_data/strategies/utils/_dup_probe.py
```

## 2. Build the outcome table

```bash
python -m research.outcomes.table --build
python -m research.outcomes.table --status
```

**Expect**: one feather per pair plus a config sidecar; `--status` prints the config hash and
the parameters behind it.

Reproducibility (SC-003) and staleness (SC-004):

```bash
python -m research.outcomes.table --verify          # rebuilds one pair, asserts identical
# then edit any exit parameter and:
python -m research.outcomes.table --status          # EXPECT: cache reported stale, not reused
```

## 3. Confirm the leakage guard (FR-009)

```bash
python -m pytest user_data/strategies/research/outcomes/tests/ -k leakage -v
```

**Expect**: feeding a frame containing `__outcome_*` columns into the label path raises.

## 4. Score the triggers

```bash
python -m research.outcomes.triggers --list
python -m research.outcomes.leaderboard --build
python -m research.outcomes.leaderboard --show
```

**Expect** a row per trigger carrying base rate, economic value with its interval, economic
value over fillable bars only, probe learnability with its spread, gap-fill share, and one of
three verdicts — all against the `empty` trigger as the null.

**The acceptance question** (SC-007): does `gbb` come out `selection_value` or `decoration`?
That is the thing this feature was built to answer. `empty` must come out `indeterminate` by
construction — it is being compared with itself, and if it does not, the verdict rule is
mis-implemented.

## 5. Sanity checks that should fail loudly

```bash
python -m research.outcomes.leaderboard --build --triggers gbb   # EXPECT ERROR: 'empty' required
```

The null cannot be omitted, because without it no verdict can be assigned.

## What this does NOT do

It produces no P&L claim. Any candidate that reaches a verdict of `selection_value` is then a
candidate for the pooled walk-forward harness in `evaluation/`, which resolves ~3–4pp/yr and
is the only acceptable source of a P&L claim (FR-017). The table's economic value is a
lookahead bound and is never a realisable return.
