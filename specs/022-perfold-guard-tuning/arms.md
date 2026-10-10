# Arms: measured results reused by feature 022

FR-014 forbids re-running A, B and D. They are recorded here so every comparison quotes the
same figures, and so arm B never travels without its provenance status (FR-002).

| arm | tag | guards | model | pooled wallet | folds + | per-trade EV | status |
|---|---|---|---|---|---|---|---|
| **A** | `wf_gbbo` | open | none | **+2.64pp** | 7/15 | +0.000155 | measured, 15 folds |
| **B** | `wf_gbbg` | globally tuned | none | **+18.76pp** | 12/15 | +0.014671 | **`in_sample` — 97.8% of it from folds inside its own tuning window. Out of sample: +0.42pp/6 folds.** See provenance.md |
| **C** | `wf_gbbgrid` | **tuned per fold** | none | — | — | — | **this feature, gated on US2** |
| **D** | `wf_aro` | open | per-fold threshold | **+39.98pp** | 15/15 | — | measured, the incumbent |
| **E** | — | tuned per fold | per-fold threshold | — | — | — | contingent on C (FR-020) |

All arms share `exit_config_hash 5a18305d1fafa83e`; equality across A/B/C/D is verified as its
own task (T032, SC-004) because a mismatch voids every comparison.

**Arm B carries a caveat until US1 completes.** Its +18.76pp came from globally tuned guard
values whose hyperopt window is unknown. If that window overlaps the 15 fold test ranges the
figure is in-sample. Separately, **198 trades across 15 folds is roughly half** the 378 the
harness needs for its 3.350 pp/yr floor, and one fold traded zero times — so it was never a
resolved result even before provenance is considered.

**Arm A is the comparator that carries the feature.** C vs A is interpretable whatever US1
returns; C vs B loses its meaning if B is in-sample.

## Test-layout deviation (T001)

tasks.md specifies `evaluation/tests/test_*.py`. The repository's actual convention is tests
**directly in** `evaluation/` (`test_pick_threshold.py`, `test_run_folds.py`), so the new tests
follow that instead: `evaluation/test_gate_is_enforced.py`, and the `guards`-key tests appended
to the existing `evaluation/test_run_folds.py` beside the `threshold` precedent they mirror. No
`evaluation/tests/` directory is created. Matching the surrounding layout beats matching a task
description written before the layout was checked.
