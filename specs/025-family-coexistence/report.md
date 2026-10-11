# Report: 025 family coexistence and reversion late-half fix

**Date**: 2026-10-11 · **Spec**: [spec.md](spec.md) · **Decisions**: [decisions.md](decisions.md) ·
**Evidence**: [baseline.md](baseline.md)

> **Every figure is PROVISIONAL (quality gate 2).** There is no distant era. Every choice was made on
> f00–f07 and judged on f08–f14 by `evaluation/stage.py`, which cannot read late folds when
> choosing (test-enforced).

## SUMMARY

1. **Nothing was adopted.** Five stages ran: coexistence, then four reversion exit steps. Each
   stage's early-fold winner was judged on the late folds against `st_fam_combo`, and none was
   IMPROVED. The final arm is therefore 024's `st_fam_combo`, unchanged. This is a negative
   result, recorded as one.

   | stage | early winner | early sum (vs +35.76) | late Δ vs `st_fam_combo` | adopted |
   |---|---|---|---|---|
   | coexistence | cap 7 pairs/family | +34.33 | −4.85pp, CI [−11.3, +5.4], p 0.50 | no |
   | tune1 take-profit | 0.015 | **+46.85** | −3.59pp, CI [−11.6, +7.2], p 0.66 | no |
   | tune2 unclog_12h | 6h | **+49.80** | +0.16pp, CI [−7.8, +8.2], p 0.99 | no |
   | tune3 ATR floor | −0.08 | +36.11 | +0.07pp, CI [−1.4, +1.6], p 0.95 | no |
   | tune4 volume-confirm | off | +32.51 | +2.83pp, CI [−8.5, +11.7], p 0.75 | no |

2. **The premise was half wrong: momentum was never crowded out in the late half.**
   - Inside 024's combined arm, the momentum side made **+4.33pp** in the late half. That is more
     than momentum alone (+2.50).
   - The combined arm's late loss is entirely its **reversion** side.
   - Contention is real. The rules cut entries lost to the other family's pair by up to 60%
     (momentum: 720/532 → 283/216 under the 48-bar window). But lost entries were not what cost
     the late half.
3. **The reversion side's late loss survives every lever tried.** Across all 15 arms, the reversion
   side's late-half contribution ranges **−14.05 to −30.95pp**. Coexistence rules change which
   pairs it gets and exit settings change how it leaves. Neither changes the sign. Together with
   023 (`gbb` alone: −19.28 late; every trigger except `gbb_tight` negative late), this points to
   the **reversion entries in the late period**, not their exits or their crowding.
4. **The recurring early/late pattern.** Tighter reversion exits (TP 0.015, 6h unclog) add 11–14pp
   in the early half and nothing late. That is the same shape as 022/023's late-half collapse. An
   early-folds-only choice would have adopted them; the late gate rejected them, as designed.
5. **Post-hoc, NOT decisions** (they read the late folds):
   - `st_coex_cap3` has the best late half (−4.45).
   - `st_tune2_u6` has the best total (+29.59).
   - Neither beats momentum alone (+25.19 total, +2.50 late) in both halves.

## SC-004 answers (with paired intervals)

| question | answer |
|---|---|
| Does the combined arm's momentum side reach momentum-alone's late half (+2.50pp)? | **Yes, already in 024**: +4.33pp in `st_fam_combo`. It ranges −2.21 to +9.60 across the 025 arms. It was never the binding problem. |
| Did entries lost to the other family fall vs 024 (3,258 reversion / 1,252 momentum)? | **Yes, under every active rule**, e.g. cap3: 2,333 / 864; win48: 3,514 / 499 (momentum down 60%, reversion up). See baseline.md. |
| Does the final combined arm beat momentum alone in each half? | **No.** Final = `st_fam_combo`. Early +35.76 vs +22.68: UNCHANGED. Late −20.37 vs +2.50: INCONCLUSIVE_ARM_LOSES. Whole span −2.66pp/yr, CI [−13.9, +8.7], p 0.65 (024). |

## All arms (per-fold wallet sums, pp; family columns are contributions inside the arm)

| arm | total | positive folds | early | late | momentum early | momentum late | reversion early | reversion late |
|---|---|---|---|---|---|---|---|---|
| `st_fam_combo` (final) | +15.39 | 8/15 | +35.76 | −20.37 | +12.31 | +4.33 | +23.45 | −24.70 |
| `st_coex_cap3` | +12.16 | 9/15 | +16.62 | −4.45 | +9.25 | +9.60 | +7.37 | −14.05 |
| `st_coex_cap5` | +1.11 | 7/15 | +22.72 | −21.61 | +10.14 | +6.30 | +12.58 | −27.91 |
| `st_coex_cap7` | +9.11 | 6/15 | +34.33 | −25.22 | +10.63 | +3.70 | +23.70 | −28.92 |
| `st_coex_win4` | +19.24 | 7/15 | +33.24 | −14.00 | +11.45 | +3.72 | +21.79 | −17.72 |
| `st_coex_win16` | +8.18 | 7/15 | +27.10 | −18.92 | +14.62 | +1.37 | +12.48 | −20.29 |
| `st_coex_win48` | +4.50 | 7/15 | +23.19 | −18.68 | +20.27 | +3.44 | +2.92 | −22.12 |
| `st_tune1_tp015` | +22.90 | 8/15 | +46.85 | −23.95 | +23.08 | −0.05 | +23.76 | −23.90 |
| `st_tune1_tp030` | +9.07 | 7/15 | +38.24 | −29.17 | +19.77 | +1.77 | +18.47 | −30.95 |
| `st_tune1_tpoff` | +7.82 | 7/15 | +30.77 | −22.96 | +10.49 | −2.21 | +20.28 | −20.75 |
| `st_tune2_u6` | +29.59 | 8/15 | +49.80 | −20.21 | +17.47 | +0.59 | +32.33 | −20.80 |
| `st_tune2_u24` | +9.64 | 6/15 | +33.77 | −24.14 | +11.49 | +4.36 | +22.28 | −28.50 |
| `st_tune3_f05` | +15.35 | 8/15 | +35.48 | −20.13 | +11.87 | +4.28 | +23.61 | −24.41 |
| `st_tune3_f08` | +15.81 | 8/15 | +36.11 | −20.30 | +11.70 | +4.03 | +24.40 | −24.32 |
| `st_tune4_vcoff` | +14.98 | 6/15 | +32.51 | −17.53 | +12.32 | +5.47 | +20.19 | −23.00 |
| *momentum alone `st_bgo_mx`* | *+25.19* | *9/15* | *+22.68* | *+2.50* | | | | |
| *reversion alone `st_gbb`* | *+11.04* | *7/15* | *+30.33* | *−19.28* | | | | |

Liveness: every active arm is LIVE (refusal counts, recency-dropped rows, or moved reversion exit
counts; see `decisions.md`). The ATR floor (tune3) is only barely live: a few exits per fold, with
10–12/15 folds differing. That confirms research R5's clamp concern for the multiplier as well.

## Success criteria

| SC | status | evidence |
|---|---|---|
| SC-001 neutral settings reproduce `st_fam_combo` | **PASS**, 15/15 both rules | baseline.md T012 |
| SC-002 active settings live, or INERT reported | **PASS**, all 14 active arms LIVE | decisions.md |
| SC-003 every choice early-made, late-judged | **PASS**, 5 stage blocks | decisions.md; `evaluation/tests/test_stage.py` |
| SC-004 questions answered with intervals | **PASS** | above |
| SC-005 023/024 fingerprints and reproduction unchanged | **PASS**: 3 fresh f00 re-runs identical; 3 fingerprints unchanged | baseline.md T024 |

## What was built

- `utils/FamilyCoexistence.py`, rule A: a per-family pair cap at entry confirmation, after the
  volume check. It is strategy-agnostic, and refusals go to the sink JSONL.
- `SimpleTrigger`, rule B: the momentum-recency guard. Overlap bars go to momentum while it is
  active. `FAMILY_EXIT_OVERRIDES` now merge along the MRO (G1).
- `regime/run_arm.py`: `coexistence_refusals`. `evaluation/fingerprint.py`: coexistence members.
- `evaluation/stage.py`: the early-choose / late-judge stage runner. It re-executes itself with
  `PYTHONHASHSEED=0`.
- 16 configs (2 neutral, 6 coexistence, 8 tuning), 16 arms × 15 folds.

## Recommended next (fix-oriented; the reversion late half is an ENTRY problem)

1. **Gate the reversion family by regime.** The late collapse recurs in every trigger and arm since
   022, which is the signature of a regime shift. A causal regime gate that switches reversion
   entries off when the gate says "trend" would test this directly. It was out of scope in
   023–025 and is now the leading lever.
2. **A more selective reversion trigger inside the combination.** `gbb_tight` was the only trigger
   positive in both halves (023), but at 89 trades. Loosening it toward `gbb` and running it as
   the reversion side would show where trade count becomes usable.
3. **Keep momentum alone as the reference to beat.** No combination here beats it in both halves.
