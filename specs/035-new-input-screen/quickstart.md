# Quickstart: 035 validation run

From `user_data/strategies`, with `set -a; source ~/.freqtrade_env; set +a; export PYTHONPATH=.`.
Commands are in [contracts/screen_cli.md](contracts/screen_cli.md).

1. **Tests:** `pytest research/new_inputs/tests -q`. The mixin is byte-identical when off, the
   alignment rules hold, the truncation harness catches the planted centred control, and the
   partial-Spearman function matches a hand-computed case.
2. **Data:** `fetch` for each source, then `build`.
   - Expect a manifest with every file verified.
   - Expect one parquet per pair from 2021-01, with positioning NaN before 2021-12-01 and its flag
     at 1 there.
3. **S0:** `causality` (every input PASS; the planted control FAILs) and `coverage`.
4. **Commit the pre-registration** (`preregistration.json` / `.md`) before `screen`.
5. **S1:** `screen`.
   - It must first print 412 trades and the control-at-share gap 0.1004 ± 0.002.
   - Then `survivors`; commit `survivors.json`.
6. **S2 (if survivors):**
   - `build_arms`, then `train --tag T` per survivor (one chain);
   - signal tables; picks and verifies (one fold per process);
   - `separation`. The passer with the largest gap above 0.1004 is the winner.
7. **S3 (if a winner):**
   - `train --tag T --folds f08..f14`, then picks;
   - `run_folds` from the repo root;
   - `instrument`, then `judge`.
8. **SC-003:** re-run `wf_aro_g24` f00 and f07 with the held-pair add declared off. The trade
   hashes must equal the records.
