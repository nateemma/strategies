# T030 — success-criterion verification

Each SC checked against the artefact that satisfies it, or recorded as NOT MET
with the reason. Verified 2026-09-06.

| SC | requirement | status | evidence |
|---|---|---|---|
| SC-001 | headroom stated with trade counts and windows | **MET** | `headroom.md`: −4.558pp W1 (155 vs 176 tr), −2.931 W2, −1.090 W3 |
| SC-002 | three-way comparison on all three windows | **MET** | `headroom.md` table: no-model / real / perfect × W1,W2,W3, all `returncode` 0 |
| SC-003 | both inherited changes resolved, config and models agree, justified on realised return | **MET** | `inherited.md`: label ADOPT 0.02, take-profit KEEP 0.022; 11 models marked STALE |
| SC-004 | every guard's contribution reported, inert guards identified | **MET** | `guards.md`: 6 guards × 3 windows; `entry_guard_threshold` INERT (identical hashes) |
| SC-005 | confidence threshold chosen by a stated decision rule, per-window optima reported | **PARTIAL — re-scoped** | The US4 *selection* task (T019/T020) was re-scoped by the T009 gate. A 5-point sweep × 4 configs × 3 windows WAS run for T016 and is reported in `inherited.md`, with the rule stated (max in-sample mean, W1 held clean). No production threshold is chosen because none is warranted: 0.4–0.6 are largely byte-identical on W1 |
| SC-006 | exit gate and exit signal each measured without confounding by the other | **NOT MET — re-scoped** | T021–T024 not executed (FR-004). The exit-rvol asymmetry and the +10.4pp exit-signal prior remain **OPEN** and are recorded as such in `NNMT_OPERATING_POINT.md` |
| SC-007 | every accepted/rejected configuration supported by ≥2 seeds with spread | **MET** | Label: 4 models, `MLX_SEED` 42/7, spread 0.475/0.813pp reported. Take-profit: both seeds, and the doc explicitly notes the identical W1 deltas are ONE measurement reported twice, not two-seed agreement |
| SC-008 | nothing accepted on a classifier metric; every ceiling labelled | **MET** | No verdict cites a metric; `val_trading_mcc` favoured the *rejected* label (0.760 vs 0.612) and was recorded as context only. Audit found no unlabelled oracle figure |
| SC-009 | no shared framework mechanism left modified | **MET** | `git status --short Framework/` clean; `NNMTStrategy.py`, `NNMT_MLX.py`, `NNMTClassifierMLX.py` untouched. The `dbg_curr_df` defect was worked around inside `NNMT_Oracle`, not fixed in shared code |
| SC-010 | conclusion sufficient to accept/reject a future proposal without new measurement | **MET** | `regime/NNMT_OPERATING_POINT.md` — six named proposal classes foreclosed, with the one non-foreclosed route stated |

## Honest summary

**8 of 10 MET, 1 PARTIAL, 1 NOT MET.** Both shortfalls are SC-005 and SC-006,
and both trace to the same cause: the T009 hard gate found no headroom, and
FR-004 required re-scoping Phase 5 rather than executing it. That is the
feature's designed behaviour on a negative gate result, not an execution
failure — but the criteria are still recorded as unmet rather than reinterpreted
as satisfied.

**What remains genuinely open**, and is the honest residue of this feature:

- The **exit-rvol asymmetry** (`exit_rvol_threshold` 2.0 against NNMT's entry
  rvol of 0.2) was never measured. Research R5 called it a first-class
  candidate. It is an EXIT-side question, so the Phase 2 headroom result does
  not speak to it (R9) — a null model contribution says nothing about how the
  exit harvests.
- The **+10.4pp exit-signal prior** at an open gate was never reproduced or
  refuted.

Both sit on the exit side, which is precisely where this feature's evidence does
NOT reach. They are the strongest remaining candidates on this family and are
recorded in the conclusion as such.
