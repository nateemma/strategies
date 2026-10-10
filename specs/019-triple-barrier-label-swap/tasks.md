# Tasks: 019 Triple-Barrier Label Swap

- [X] T001 [MI] Strict triple-barrier labeler fix in `Framework/TrainingSignals.py` + tests (cf6abf5)
- [X] T002 [MI] `label` field in `research/clfstudy/registry.py` / `labels.py` / `data.py` / `run.py` (64ee46b)
- [X] T003 [MI] `--ridge-phase` option in `research/clfstudy/gate.py` (ridge reference restricted to that phase's RIDGE rows) + test
- [X] T004 [MAIN] `research/clfstudy/configs/phase_19.yaml` per FR-003; verify ids and class balance
- [X] T005 [ER/MAIN] Smoke test (`python -m research.clfstudy.smoke`) must pass, then `phase --phase 19`
- [X] T006 [ER] `leaderboard --phase 19`; `gate --phase 19 --ridge-phase 19`
- [X] T007 [GR] Review memo `gates/review_19.md`
- [X] T008 [MAIN] Checkpoint with the author; decision: screening campaign (ATR scales -> raw OHLCV -> horizon x width -> window length) with RIDGE x3 + MLP-64 x2 and the 0.02 / 0.05 promotion rule
- [ ] T009 [MAIN] Report section + commit
