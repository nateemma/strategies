# Quickstart — 005 NNMT Operating Point

Run from the repository root. Nothing here trains a model except where stated.

## Window treatment (applies throughout)

W1 `20250901-20260831` is the SOLE acceptance window. W2 `20240601-20250531`
and W3 `20230101-20231231` are in-sample and are reported, never accepted on.
**W3 is additionally near-flat on this family**, so any W3-based statement
carries its trade count and an explicit flag.

Arms run SEQUENTIALLY. Two seeds (via `MLX_SEED`) before any acceptance —
`train_seed` is inert for multi-task and would silently produce identical models.

Guards-off is always the six neutralising values, never the guard flag:

    NEUT="--set entry_rvol_threshold=-1.0 --set entry_atr_pct=-1.0 \
          --set entry_adx_threshold=-1.0 --set entry_bb_width_threshold=-1.0 \
          --set entry_guard_threshold=5.0 --set entry_close_norm_threshold=5.0"

## Scenario 1 — the control (Phase 1)

    PYTHONPATH=user_data/strategies .venv/bin/python \
      user_data/strategies/regime/run_arm.py --strategy NNMT_MLX_Base \
      --window W1 --tag nnmt_ctl

**Expected**: `returncode` 0 and a recorded return. Confirm the model on disk was
trained at the control's label; if not, retrain once before proceeding.

## Scenario 2 — headroom (Phase 2, the hard gate)

Three arms at the control configuration: `NNMT_NoModel`, `NNMT_MLX_Base`,
`NNMT_Oracle`, each on all three windows.

**Expected**: a three-way comparison with trade counts. The gate question is the
perfect-minus-real difference on W1.

**If that difference is not materially larger than the resolution implied by the
observed trade count, STOP and re-scope.** State the power threshold as a
function of the observed count — do not assume the 1pp floor calibrated for
~97-trade arms.

## Scenario 3 — guard attribution (Phase 3)

Neutralise each guard in turn at the control configuration.

**Expected**: per-guard delta and trade count. A guard changing neither is
reported as INERT. The sum of individual deltas is reported against the joint
effect.

## Scenario 4 — the inherited changes (Phase 4)

Label and take-profit introduced **separately**, each against the control, each
on two seeds, each with its own re-swept operating point.

**Expected**: two independent verdicts. The label arm must pin the profit-head
threshold so it stays single-variable.

## Scenario 5 — the exit signal (Phase 5)

Measured on the REAL model, at both the chosen exit-rvol value and fully open.

**Expected**: a result that cannot be an artefact of the exit gate. An oracle
null is not evidence here — the gate bounds what a label can pay, not how the
exit harvests it.

## Scenario 6 — hygiene

    git status --short Framework/    # must be clean

**Expected**: no shared mechanism modified; every ceiling figure labelled; no
oracle configuration promoted.

## Success criteria coverage

| SC | validated by |
|---|---|
| SC-001, SC-002 | Scenario 2 |
| SC-003 | Scenario 4 |
| SC-004 | Scenario 3 |
| SC-005 | Scenario 5 (threshold sweep) |
| SC-006 | Scenario 5 |
| SC-007 | Scenarios 4 and 5 (two seeds) |
| SC-008 | Scenarios 2 and 6 |
| SC-009 | Scenario 6 |
| SC-010 | Phase 6 conclusion |
