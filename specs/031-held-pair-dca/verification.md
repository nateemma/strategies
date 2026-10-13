# 031 T016 — switch off: byte-identical (FR-007, SC-001)

The 7 reference arms were re-run with the 031 code under `_c031` tags: plans copied with only
`run_id`/`arm_tag` renamed, so classes, params snapshots, thresholds and model dirs are the same.
One sequential chain, 15:11–15:37 on 2026-10-13. `baseline.py --check baseline.json --suffix
_c031`:

| reference arm | per-fold (n_trades, trades_hash) equal | framework_sha | model key |
|---|---|---|---|
| `wf_aro_b030` (arm D, 6.5 h) | 15/15 | unchanged | unchanged |
| `wf_aro_g24` | 15/15 | unchanged | unchanged |
| `fs_gbr_gr` | 15/15 | unchanged | unchanged |
| `fs_gfm_gr` | 15/15 | unchanged | unchanged |
| `fs_mlx_gr` | 15/15 | unchanged | unchanged |
| `st_bgo_mx` | 15/15 | unchanged | n/a (no model) |
| `st_prom_c2` | 15/15 | unchanged | n/a (no model) |

Also checked:
- the 8 base classes' framework_sha are unchanged;
- no `_c031` record carries `n_adds` or `hp_counters`;
- every NNNC fold logged "model already present — skipping training", and no new weight file
  exists under `saved_data/`.

**Result: ALL EQUAL (105/105 folds).**

**Operational note.** Two plans that share fold classes (`wf_aro_b030` and `wf_aro_g24` both use
`NNNC_ARO_fNN`) rewrite those classes' `.json` from their own snapshot. While the chain is running
`wf_aro_b030`, anything that reads `NNNC_ARO_fNN.json` sees 6.5 h grace. Two things hit this:
- `evaluation/tests/test_fingerprint.py::test_fold_class_shares_arm_d_framework` failed
  transiently. It passed again once `wf_aro_g24` had restored the 24 h state;
- the ARO screen's book-rebuild fidelity block was corrupted while it ran in parallel. The file
  was restored from git; the policy numbers set grace explicitly and were unaffected.

Do not run replays or tests that read the fold `.json` while a chain runs.
