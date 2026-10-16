# 036 Realistic P&L for NNNC: report

Spec: `specs/036-realistic-nnnc-pnl/`. **Label: PROVISIONAL.**
- Fold-based outputs (re-statement, capacity) are execution measurements, not edge findings. They
  run on fold models that inherit the full-history scaler (FR-020; plan.md records the Principle XI
  exception).
- Promotion of anything is the user's decision.

## Setup (2026-10-16)

- **Recorded live data** goes to `user_data/data/live_nnnc/`. That path is outside the nested
  strategies repository, and the outer upstream checkout ignores it (`.gitignore: user_data/*`).
  It is never committed (FR-008).
- **The live bot's database**, `tradesv3.live_nnnc.sqlite`, is at the repo root and ignored there
  (`.gitignore: *.sqlite`). It is not in the strategies repository.
- **The momentum dry-run** (pid 2661, port 8080, `tradesv3.dryrun.sqlite`,
  `config_mom_dryrun_15m.json`) is out of bounds for this feature.
- **Tests:** AGENT_GUIDE's suite line uses the generic `research/<study>/tests`, which covers
  `research/live_pnl/tests`, so no edit was needed.

## Phase 2 checkpoint (2026-10-16)

- **Tests:** the full suite passes (1122 passed, 9 skipped), including 19 new `research/live_pnl`
  tests.
- **Existing arms:** byte identity holds by construction. Phase 2 adds only new files
  (`utils/ExecutionRecorder.py`, `research/live_pnl/*`, `config/config_live_nnnc.json`,
  `scripts/live_nnnc.sh`) and edits no existing code, so no existing class imports or inherits
  anything new. The recorder's backtest pass-through and key neutrality are tested directly
  (`test_recorder.py`).
- **Launcher:** `live_nnnc.sh --check` correctly refuses before a live model exists.
