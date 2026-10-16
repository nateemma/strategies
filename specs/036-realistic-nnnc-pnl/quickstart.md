# Quickstart: Realistic P&L for NNNC (036)

Paths are relative to `user_data/strategies` unless a step says "repo root". Source the
environment first: `set -a; source ~/.freqtrade_env; set +a`.

## 1. Live model (US1)

```sh
# refresh 15m data for the 10 pairs (repo root)
.venv/bin/freqtrade download-data -c user_data/strategies/config/config_wf.json -t 15m --days 10
# train + pick + write the _REC class (one trainer at a time)
PYTHONPATH=. ../../.venv/bin/python -m research.live_pnl.live_fold --cut <YYYYMMDD> --run-id live1
```

**Expect:**
- `research/live_pnl/live_models.json` has `live1` with its key, threshold and train range;
- `scripts/model_index.py key NNNC_LIVE_live1_REC` equals the recorded key.

## 2. Start the bot

```sh
zsh scripts/live_nnnc.sh start        # repo root
zsh scripts/live_nnnc.sh status
```

**Expect:**
- the bot listens on port 8081 and writes to `tradesv3.live_nnnc.sqlite`;
- the momentum bot (pid 2661, port 8080) is unchanged: check with `ps` and its API;
- within 15 minutes, `user_data/data/live_nnnc/live1/depth-<today>.jsonl.gz` has one scheduled
  snapshot per pair, and `signals-<today>.jsonl` has one row per pair.

## 3. Daily and weekly

```sh
PYTHONPATH=. ../../.venv/bin/python -m research.live_pnl.parity --day <yesterday>
PYTHONPATH=. ../../.venv/bin/python -m research.live_pnl.drift      # weekly
PYTHONPATH=. ../../.venv/bin/python -m research.live_pnl.gate
```

**Expect:**
- `unexplained` is 0 on parity reports, or the difference is investigated before any P&L read;
- drift stays within ×0.5–×2 of 0.014735.

## 4. After the gate opens (US2, US3)

```sh
PYTHONPATH=. ../../.venv/bin/python -m research.live_pnl.cost_model --fit
PYTHONPATH=. ../../.venv/bin/python -m research.live_pnl.restate --arm wf_aro_g24
PYTHONPATH=. ../../.venv/bin/python -m research.live_pnl.restate --arm w_aro
# capacity: pre-register budgets, run the arms (repo root, one chain), then judge
PYTHONPATH=. ../../.venv/bin/python -m research.live_pnl.capacity
```

**Expect:**
- per-pair held-out error is within tolerance, or the pair is listed as excluded;
- the restated baselines have gross, cost and net per fold;
- `capacity.json` has a verdict per budget, and every on-arm is live (realised stakes differ from
  the reference).

## 5. Shared wallet (US4, last)

Backtest the momentum basket on the fold ranges, then run
`evaluation/books.py --shared-wallet` with `cap_aro_ref`. The combined P&L is reported against the
sum of the two, as deployment information only.

## Checks that must hold throughout

- the full test suite passes;
- with every new option off, existing arms are byte-identical (trades hash);
- `git status` shows no files from `user_data/data/live_nnnc/`.
