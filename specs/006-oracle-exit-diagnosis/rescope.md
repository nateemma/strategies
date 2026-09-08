# RE-SCOPE — the founding paradox dissolved, 2026-09-07

This feature was convened because a perfect-entry arm earned LESS than the real
model. **That is no longer true.** T007 found the cause — the arm was entering up
to two bars early on dilated training labels — and after the fix:

| NNMT W1 | trades | return |
|---|---|---|
| real model | 176 | +10.708% |
| **perfect entry (corrected)** | 136 | **+11.809%** |

The premise is gone. This document records what that costs the feature, what
survives, and what the feature becomes.

## What the feature already delivered — and it was the valuable part

**US1 (instrument validation) is COMPLETE and was worth the whole feature.** It
found a defect that had silently biased **every oracle figure in this
repository** across features 004, 005 and 006, produced the canonical corrected
record (`regime/ORACLE_CORRECTION_2026-09-07.md`), forced the rewrite of a
document whose central thesis was an explanation of the bug, and refuted one
conclusion I had given as an argument against spending a retrain.

That was scoped as a BLOCKING precondition. It turned out to be the deliverable.

## What DISSOLVED

**US2's motivating question.** "How much is the exit leaving on the table, given
that perfect entries lose?" — perfect entries do not lose. The exit is not
implicated by a paradox any more; it is an ordinary open question with ordinary
expected value.

**The urgency of US3 and US4.** Both were contingent on a large bound. With
headroom at the gate now only +1.101pp on NNMT, the exit cannot be hiding much
at that gate: the entire distance between the real model and a PERFECT ENTRY is
about one point.

## What SURVIVES, and is still worth doing

**The bound itself (US2) is still worth one arm.** It is cheap, it is now
measurable on a trustworthy instrument, and it answers a question no corrected
figure has answered: *of the +1.101pp between the real model and a perfect entry,
how much does the exit stack consume?* A patient exit could also earn MORE than
the ceiling implies, since the ceiling is measured under the CURRENT exit.

**The payoff geometry (T008-T009) is still worth computing.** NNMT risks 2–4% to
harvest 2.2% on a move guaranteed to reach only 2.0%. That arithmetic is
independent of the instrument defect and remains the cheapest thing in the
feature.

**US5's discipline survives intact** and its warning is now doubly earned: an
instrument-only result is exactly what this feature has just been burned by.

## What the correction OPENED, and it is bigger than the exit

Corrected headroom at the gates the families actually run:

| family | real model | ceiling (`gbb`) | headroom | ceiling (`gbb_path`) | headroom |
|---|---|---|---|---|---|
| NNMT | +10.708% | +11.809% | **+1.101pp** | +14.248% | **+3.540pp** |
| NNNC | +10.446% | +13.914% | **+3.468pp** | +15.090% | **+4.644pp** |

**Feature 005 concluded NNMT had NEGATIVE headroom and foreclosed model-side
work on that basis. Corrected, headroom is positive on both families** — and
under the path label it is +3.5pp and +4.6pp. Meanwhile the retrained `gbb_path`
models captured **−0.091pp** (NNMT) and **+0.747pp** (NNNC, noise).

So the live question is no longer "is there room?" — there is — but **"why does a
trained model capture none of a 3–5pp gap?"** That is a larger and better-posed
question than the exit one, and it is NOT this feature's scope.

## The re-scoped plan

| story | status |
|---|---|
| US1 instrument | **DONE.** Cause found, fixed, 9.6% residual bound carried |
| US2 geometry + bound | **RUN, REDUCED.** T008-T009 (geometry) and T010-T012 (one patient arm). Expect a small bound |
| US3 attribution | **CONTINGENT and now unlikely.** Only if the bound exceeds its resolution — with ~1.1pp of total headroom, it probably will not |
| US4 ordering | **CONTINGENT.** Same condition |
| US5 real-model confirmation | **UNCHANGED.** Still the only promotion path; nothing promotes without it |
| US6 conclusion | **RUN, REFRAMED.** Records the instrument finding as the primary result and the exit bound as secondary |

**Budget after re-scope**: ~4 arms and zero retrains, against the ~30 originally
planned.

**FR-005 is the authority for this.** It required exactly this response — state
that the exit is not the constraint and re-scope rather than execute the
attribution as written. The trigger arrived earlier than expected and from a
different direction, but the rule is the one being followed.

## Recommendation for what comes next

The exit bound should be finished because it is nearly free. But the honest
priority order after that is:

1. **Correct feature 005's foreclosure.** `NNMT_OPERATING_POINT.md` forecloses
   model-side work on a headroom figure that is now positive. It carries a
   banner, but the conclusion needs re-deriving on corrected numbers.
2. **Ask why a 3–5pp corrected gap is unreachable.** Both families now show real
   headroom under the path label and no trained model has captured any of it.
   That is the successor question, and it deserves its own feature rather than
   being smuggled into this one.
