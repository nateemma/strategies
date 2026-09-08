# T012 — the exit's opportunity cost. **THE EXIT IS NOT THE CONSTRAINT.**

**Verdict: FR-005's condition is met. The bound is NOT materially larger than
its resolution on the acceptance window — it is NEGATIVE. Phases 4-6 are
re-scoped, not executed.**

## The measurement

Perfect entries (corrected instrument, label 0.02) under the current exit versus
a patient exit held to the label's own 12-hour horizon: ATR stop off, ROI off,
take-profit lifted clear, `unclog_12h` deferred via the existing HoldWindowMixin.

| window | current exit | PATIENT exit | bound | paired 2sd | net/gross | verdict |
|---|---|---|---|---|---|---|
| **W1** (acceptance) | 136tr +11.809% | 100tr +8.448% | **−3.361pp** | 8.37 | 30.5% | within noise |
| W2 (in-sample) | 285tr +13.458% | 201tr +33.066% | +19.608pp | 11.32 | 60.1% | RESOLVED |
| W3 (in-sample) | 93tr +7.517% | 85tr +10.098% | +2.581pp | 2.95 | 75.4% | within noise |

**On the acceptance window a patient exit is WORSE, and not resolvably so.** The
exit stack is not withholding return from perfect entries; if anything it is
adding some. Waiting for the promised move means sitting through the drawdown on
the way, and the ATR stop was banking more than it cost.

## A hypothesis I raised and then REFUTED

The patient arm takes 26% fewer trades (100 against 136), which looked like slot
contention: it holds 52 bars against 11, and peak concurrency read 10/10 —
apparently saturating `max_open_trades`. That would have confounded the exit
change with a smaller entry set.

**Tested by raising the limit to 40 on BOTH arms. The hypothesis is refuted:**

| | limit 10 | limit 40 |
|---|---|---|
| patient trades (W1) | 100 | **100** — identical |
| peak concurrency | 10/10 | **10/40** |

Peak stays at 10 with four times the room, so the limit was never binding and the
10/10 reading was coincidence. The reduction is **per-pair occupancy** —
freqtrade will not re-enter a pair it already holds, and holding ~5x longer means
more signals arrive while that pair is occupied.

**That confound cannot be removed, because it IS the intervention.** Being
patient necessarily forgoes entries. Any exit change that lengthens holds pays
this cost, and the bound above already includes it — which is the honest way to
price patience.

## W2 disagrees loudly, and is not promoted to a verdict

+19.608pp on W2 is RESOLVED against its own band, and it is the largest exit-side
effect measured in this programme. It does not change the verdict:

- W2 is **in-sample** and is not an acceptance basis (FR-014).
- W1 and W2 disagree in SIGN, which is the signature of a regime-specific effect
  rather than a mechanism — the same pattern that made W3's three consecutive
  sign-flips uninformative.
- The direction is at least coherent: W2's net/gross is 60.1% under the patient
  exit against 30.5% on W1, so W2 is a window where holding for the move paid.

**Recorded as an open observation, not a finding.** If the exit line is ever
reopened, W2 is where to look — but a change cannot be accepted on an in-sample
window that contradicts the acceptance window.

## Why this was predictable from the geometry (T008-T009)

| family | median actual stop | harvest | reward : risk | breach own stop before target |
|---|---|---|---|---|
| NNMT | −2.58% | +2.2% | **0.85 : 1** | **16.3%** |
| NNNC | −4.15% | +2.4% | **0.58 : 1** | 5.8% |

Both families risk more than they harvest even on guaranteed moves. But that is
survivable, not broken: the corrected oracle wins 64.7% of its trades, so
0.65 x 2.2% − 0.35 x 2.58% ≈ **+0.53% per trade**, and it returns +11.809%.

Note the cross-family tension. NNNC has the WORSE ratio but breaches its stop
**3x less often** (5.8% against 16.3%), because its wider band (−4.15% against
−2.58%) gives the move room to arrive. Tightening the stop improves the ratio and
costs breaches; loosening does the reverse. **NNMT sits on the tight side of that
trade-off** — which is the one concrete exit-side observation this feature
produces, and it is NOT actionable without a real-model test that Phase 6 would
now be re-scoped out of.

## FR-005 disposition

The bound on W1 is −3.361pp against an 8.37pp resolution. **The exit is not the
constraint on perfect entries.** Per FR-005 and the T012 gate:

- **Phase 4 (attribution) — RE-SCOPED, not executed.** There is no positive gap
  to attribute to a mechanism.
- **Phase 5 (target/harvest ordering) — RE-SCOPED.** Same condition. The
  geometry table above already carries the useful part of what it would have
  measured.
- **Phase 6 (real-model confirmation) — RE-SCOPED.** Nothing survived to confirm.
- **Phases 7-8 still run**: the conclusion and the hygiene checks.

This is the second time this feature has re-scoped, and both times the trigger
was a measurement rather than a judgement call. The first (the instrument defect)
removed the feature's premise; this one removes its remaining hypothesis.
