# Tasks: 019 Triple-Barrier Label Swap

- [X] T001 [MI] Strict triple-barrier labeler fix in `Framework/TrainingSignals.py` + tests (cf6abf5)
- [X] T002 [MI] `label` field in `research/clfstudy/registry.py` / `labels.py` / `data.py` / `run.py` (64ee46b)
- [ ] T003 [MI] `--ridge-phase` option in `research/clfstudy/gate.py` (ridge reference restricted to that phase's RIDGE rows) + test
- [ ] T004 [MAIN] `research/clfstudy/configs/phase_19.yaml` per FR-003; verify ids and class balance
- [ ] T005 [ER/MAIN] Smoke test (`python -m research.clfstudy.smoke`) must pass, then `phase --phase 19`
- [ ] T006 [ER] `leaderboard --phase 19`; `gate --phase 19 --ridge-phase 19`
- [ ] T007 [GR] Review memo `gates/review_19.md`
- [ ] T008 [MAIN] Checkpoint with the author; record the decision on ATR-scaled / other-horizon variants
- [ ] T009 [MAIN] Report section + commit
