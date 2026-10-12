# Quickstart: Target-Aware Exits (027)

Run from `user_data/strategies` with `../../.venv/bin/python`. Source `~/.freqtrade_env` for
backtests. One backtest chain at a time.

## 1. Feasibility (US1, no backtests)

```
python -m research.exit_coupling.feasibility --momentum st_bgo_mx --reversion st_book_rev2 \
    --out specs/027-target-aware-exits/feasibility.md
```

Expected:
- (a)–(d) per family × half;
- per coupling: the rule, its values, and SUPPORTED / NOT SUPPORTED;
- exit 3 means STOP (record the negative in `report.md` and finish).

## 2. Switch-off reproduction (US2)

```
python -m pytest -q utils/tests evaluation/tests SimpleTrigger
python -m evaluation.fingerprint --strategy SimpleTrigger_RevOnlyVolreg2   # sha unchanged
python -m evaluation.run_folds --plan evaluation/folds/plan_st_cpl_off_rev.json   # and _off_mom
python -m evaluation.coupling_liveness --arm st_cpl_off_rev --ref st_book_rev2
```

Expected: unchanged shas; every fold's `trades_hash` equal to the reference arm's; liveness 0.

US1 must first pass its null-replay fidelity gate (≥ 90% of recorded exits reproduced per
family), or it exits 2 with no verdict.

## 3. Coupling arms (US3)

```
python -m evaluation.run_folds --plan evaluation/folds/plan_st_cpl_<ref>_c<n>.json
PYTHONHASHSEED=0 python -m evaluation.halves --arm <pooled arm> --control <pooled ref>
```

Expected: non-zero `evaluation.coupling_liveness` counts in each half; late P&L verdict and risk
line, recorded in `decisions.md`.

## 4. Report (US4)

`report.md`: the feasibility table, switches built, per-family verdicts by half, the
recommended NNNC exit. Every figure labelled PROVISIONAL.
