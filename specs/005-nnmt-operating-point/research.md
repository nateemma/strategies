# Phase 0 Research — 005 NNMT Operating Point

All items resolved. Nothing carried into Phase 1 as NEEDS CLARIFICATION.

---

## R1. The control problem — DECIDED: pin a clean pre-inheritance control

**Decision**: option (b). Build a control at label 0.007 / take-profit 0.032 —
NNMT's configuration before the NNNC work touched it — and introduce the two
inherited changes one at a time.

"NNMT as it stands today" is not a valid control for anything: its models on
disk were trained at label 0.007 while the class now says 0.020, and its
take-profit moved 0.032 → 0.022. Measuring that mixture would confound a stale
model, a changed label and a changed exit in a single number.

**Cost**: one extra arm set, plus one retrain if the chosen label differs from
the models on disk. Against the alternative — a headline number nobody can
attribute — that is cheap.

**Alternatives considered**: measuring as-is and "noting" the confound. Rejected:
this feature exists partly because an unnoticed inheritance changed a family's
configuration, and repeating that pattern in the measurement would be perverse.

---

## R2. The label change reaches FURTHER on NNMT than it did on NNNC

`MIN_BUY_GAIN_THRESHOLD` is read in `NNMTStrategy` at two additional sites
(~438, ~510) where it sets the **profit head's** class boundaries at
`PROFIT_HORIZON` — not the buy/sell labels, and not at `HORIZON`.

So on NNMT the inherited label change alters **an auxiliary task's targets as
well as the main ones**. On NNNC it altered only the buy/sell labels.

**Consequence for the plan**: the label arm is not a single-parameter change on
this family. Either the profit-head threshold is pinned to its old value so the
label change is isolated, or the coupling is reported as part of the arm. The
plan takes the first option, because Constitution VIII is precisely about not
moving two things at once.

**Note**: the profit head is one of `_FILTER_TASKS` and may carry zero loss
weight, in which case its labels are computed but untrained. That must be
checked before deciding whether the coupling matters in practice — if the head
is untrained the coupling is cosmetic.

---

## R3. The oracle pattern transplants unchanged

NNMT does **not** define `get_entry_conditions`; it inherits BaseNNStrategy's,
which calls `get_predictions` then `process_predictions`. NNMT overrides those
two to consume its six-task prediction dict and set `predict_buy`/`predict_sell`.

An oracle that overrides `get_entry_conditions` therefore bypasses the entire
multi-task predict path and sets those columns directly — exactly as
`NNNC_Oracle` does. **The NNNC pattern transplants without modification.**

It also side-steps `apply_task_filters` and the forward-peak-regressor filter,
which is correct: the oracle is measuring what a perfect ENTRY signal is worth,
not what the task machinery does.

---

## R4. NNMT has no parameter file — every arm must create one

`NNMT_MLX` inherits class defaults directly. FR-022 requires each arm to carry
its own file, and on this family that is not a formality: without one, an arm
silently tracks whatever the shared defaults become. The inheritance that
motivated this feature happened through exactly that route.

---

## R5. The exit-side asymmetry is real but is NOT a recent change

`exit_rvol_threshold` is 2.0 on NNMT while its own entry relative-volume
threshold is 0.25 — eight times looser. That asymmetry has existed for the life
of the family, because the exit value was a hardcoded literal until 2026-09-06.
Parameterising it changed its tunability, not its value.

So this is a long-standing untested asymmetry newly made testable, not a
regression. The plan treats it as a first-class candidate rather than a
correction.

---

## R6. The power rule cannot be set in advance

`regime/tail/compare.py` uses a 1.0pp floor calibrated for ~97-trade arms. NNMT's
trade counts at its own gate are unknown. The plan therefore defers the power
statement until the baseline is measured, and requires the threshold to be
stated as a function of the observed count rather than assumed.

Do not apply the 1pp floor to a high-count arm: on NNNC's guards-off arms
(1760-6015 trades) that would have called noise a result.

---

## R7. NNMT's third window needs more care than NNNC's did

NNMT's W3 has been near-flat at every gate previously tested. A conclusion that
rests on a W3 movement is weaker on this family than the same movement would be
on NNNC. The plan requires W3 findings to be reported with their trade count and
flagged where they carry a conclusion.

---

## R8. Seeds

`train_seed` is inert for multi-task; the `MLX_SEED` environment variable is the
live control. Two seeds minimum before any acceptance or rejection (FR-020).

Recorded because a `train_seed` sweep on this family would silently produce
identical models and look like a null.

---

## R9. What the oracle gate can and cannot settle

The gate bounds what a LABEL can pay. It does **not** bound how the EXIT should
harvest. Demonstrated 2026-09-06: changing the take-profit was worth +0.91pp at
the ceiling but 0.5-5.6pp on the real model, because a perfect model only enters
bars that reach the target while a real one does not.

**Consequence**: exit-side questions (US5, US6) MUST be measured on the real
model. An oracle null is not evidence against an exit change.

---

## Summary of decisions

| id | decision |
|---|---|
| R1 | Pin a clean pre-inheritance control; introduce the two changes one at a time |
| R2 | The label also drives the PROFIT head on NNMT; pin that head's threshold to isolate the change |
| R3 | The NNNC oracle/bypass pattern transplants unchanged |
| R4 | Every arm creates its own parameter file; NNMT has none today |
| R5 | The exit-rvol asymmetry is long-standing, newly testable, first-class |
| R6 | Defer the power threshold until the baseline trade count is known |
| R7 | W3 conclusions need their trade count and an explicit flag |
| R8 | Seeds via MLX_SEED, two minimum |
| R9 | Exit-side questions are measured on the real model, never settled by an oracle null |
