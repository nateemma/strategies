# Report: 024 trigger-family exit presets

**Date**: 2026-10-11 · **Spec**: [spec.md](spec.md) · **Evidence**: [baseline.md](baseline.md)

> **Every delta in this report is PROVISIONAL, not a finding (quality gate 2).** There is no distant
> era; persistence is tested by the early/late halves split only (f00–f07 / f08–f14).

## SUMMARY

1. **The mechanism works and is exact.**
   - A trade's exits are now chosen by its entry family, which is tagged at entry
     (`"<family>:<trigger>[+ov]"`) and read by one framework hook,
     `BaseStrategy._exit_params(trade)`.
   - The hook is provably neutral. Four recorded arms re-run fresh give identical trades, arm D
     included (T005).
   - Family arms reproduce 023 exactly: `gbb` gives `st_gbb`, and `breakout_gbb` gives `st_bgo_mx`,
     15/15 folds each (SC-001).
   - Each side of the combined arm reproduces its single-family arm, 15/15 (SC-002).
2. **Combining `gbb` (reversion exits) with `breakout_gbb` (momentum exits) does not add up.**

   | arm | per-fold sum | positive folds | early | late |
   |---|---|---|---|---|
   | combined `st_fam_combo` | **+15.39** | 8/15 | +35.76 | **−20.37** |
   | reversion alone `st_gbb` | +11.04 | 7/15 | +30.33 | −19.28 |
   | momentum alone `st_bgo_mx` | **+25.19** | 9/15 | +22.68 | **+2.50** |
   | sum of the two singles (reference, not a tradeable arm) | +36.23 | 9/15 | +53.01 | −16.78 |

   - The combined arm lands between its two parts and well below their sum.
   - It is not resolved against either single arm. Against `st_gbb`: +1.18pp/yr, CI [−11.1, +13.4],
     p 0.85. Against `st_bgo_mx`: −2.66pp/yr, CI [−13.9, +8.7], p 0.65.
   - In the late half it loses money where momentum alone made money.
3. **Why: the families block each other at the pair level, never at the slot level.**
   - `slots_full` = **0** in every cell. `max_open_trades` is 10 and there are ≤10 pairs, so the
     slot cap never binds.
   - All contention is one-position-per-pair. A reversion trade sitting on a pair blocks a momentum
     entry on that pair, and vice versa:

   | family | half | single-arm trades | in combined | lost | lost because the other family held the pair | gained |
   |---|---|---|---|---|---|---|
   | reversion | early | 5,027 | 3,462 | 2,313 | **1,939** | 748 |
   | reversion | late | 4,029 | 2,938 | 1,583 | **1,319** | 492 |
   | momentum | early | 3,411 | 2,951 | 1,066 | **720** | 606 |
   | momentum | late | 2,745 | 2,376 | 787 | **532** | 418 |

   - Inside the combined arm, the momentum side still earns in both halves (+12.31 / **+4.33** pp).
   - The reversion side carries `gbb`'s known late-half collapse (+23.45 / **−24.70** pp). It also
     takes pairs that momentum would have used.
   - The momentum entries lost to reversion-held pairs summed to +400 profit-ratio-% in
     `st_bgo_mx`.
4. **Overlap bars are a non-issue, as 023 measured.** There were 239 `+ov` entries (137 early,
   102 late) out of 11,727, all resolved to reversion by priority.
5. The per-fold correlation between the families is **+0.54** (constitution IX). They diversify
   partly, not orthogonally.

## Success criteria

| SC | status | evidence |
|---|---|---|
| SC-001 family arms reproduce 023 | **PASS**: 15/15 + 15/15; exit counts identical; 0 untagged | baseline.md T018 |
| SC-002 one side disabled reproduces the other | **PASS**: 15/15 + 15/15 | baseline.md T022 |
| SC-003 fingerprint covers family sets / mapping / priority; arm D unchanged | **PASS** | `evaluation/tests/test_fingerprint.py`; arm D and `SimpleTrigger_Gbb` both `303456c1fa322b4c` |
| SC-004 dry-run dump identical | **PASS** (and the dry-run never imports `BaseStrategy`, research R9) | baseline.md T005 |
| SC-005 combined report by half, paired verdicts, correlation, all PROVISIONAL | **PASS** | this report; `evaluation/contention_st_fam_combo.{json,md}` |
| SC-006 every trigger has a family; test-enforced | **PASS**: 24/24 | `research/outcomes/tests/test_trigger_families.py` |
| SC-007 zero untagged trades in any SimpleTrigger run | **PASS**: 0 across 5 family arms × 15 folds | arm records `untagged_trades` |

## What was built

- **`Framework/BaseStrategy.py` (C1).**
  - `_exit_params(trade=None)`, with `trade` passed at all three call sites.
  - The stop path reads the adaptive flag, volume-confirm and static stop from params (R2).
  - It contains no family logic.
- **`utils/FamilyExits.py` (C2).**
  - A strategy-agnostic `FamilyExitMixin`.
  - Class-creation checks: forbidden shared keys, F3 ambiguity, F5 hold-window sanity.
  - `class_level_exit_params`.
- **`utils/StopInvariants.py` (C3):** the bound is read from the trade's own params.
- **Trigger registry (C5):** `Trigger.family` / `family_reason` on all 24 triggers, and a `--list`
  column. Causality verdicts are unchanged.
- **`SimpleTrigger` (C4).**
  - The `ARM_D_FAMILY` preset.
  - `TRIGGERS` / `DISABLED_TRIGGERS` / `FAMILY_PRIORITY` / `FAMILY_EXIT_OVERRIDES`.
  - Family tagging with `+ov`.
  - Configs `FamGbb`, `FamBgo`, `FamCombined`, `FamCombinedRevOnly` and `FamCombinedMomOnly`.
- **`evaluation/fingerprint.py` (C7):** family members, hashed only when used (F1).
- **`evaluation/contention.py` (C6).** `pool.py` now carries `enter_tag`.
- **`regime/run_arm.py` (C8):** `untagged_trades`, per-family trade and exit counts, and overlap
  entries, all from the archive.
- **`evaluation/run_folds.py`:** the causality preflight checks every enabled trigger of a combined
  arm.

## Deviations and defects found

- **The spec's premise about the dry-run was wrong (R9).** `MomentumRegimeBasket15mFast` derives
  from `IStrategy` and imports nothing from `Framework/`. 023 T032 had already recorded this.
- **The first T005 attempt was not a test.** `run_folds` skips folds whose record exists, so the
  gate was re-run via `run_arm` under fresh tags.
- **Class-as-self `_exit_params` read hyperopt Parameter defaults** (`cexit_take_profit` 0.008), not
  `sell_params` (0.022). This was caught by the fingerprint test and fixed with
  `class_level_exit_params`. Live instances were never affected.
- **`run_folds` refused combined arms** because it read only `TRIGGER_NAME`. Fixed, with a test.
- **Ten 023 `_MomExit` configs were never allowlisted** in the exit single-source test, which is a
  023 oversight. They are now classified as CONFIGURING.
- **Subclass per-family overrides use `FAMILY_EXIT_OVERRIDES`**, not a `FAMILY_EXITS` merge (data
  model). Declaring `FAMILY_EXITS` directly is refused, the same rule as `buy_params`.
- **The quickstart's and T001's `dump_strategy --config` flag did not exist.** Corrected.

## Scope for the deferred tuning feature (FR-016)

The contention result moves the lever. Exit parameters are not what limits the combined arm.
Pair-level blocking and the reversion side's late half are. Options, all fix-oriented:

1. **Let the families stop blocking each other per pair.**
   - For example, a per-family position budget, so a family can't occupy more than k of the
     pairs.
   - Or an entry rule that does not let a reversion trade start on a pair while momentum
     conditions hold.
   - Both are structural, not per-pair (constitution X).
2. **Make the reversion side earn its pairs in the late half.**
   - Use a stricter reversion trigger inside the combination (023: `gbb_tight` was the only
     trigger positive in both halves, but at 89 trades).
   - Or tune the reversion exit set first: one parameter at a time, early folds to choose, late
     folds to judge (FR-012).
3. **Measure the momentum side uncrowded.** This needs option 1. Reversing `FAMILY_PRIORITY`
   alone would NOT do it: priority decides only same-bar overlap (239 entries), not which family
   already holds a pair. The test is whether momentum keeps its standalone +2.50 late half once
   reversion cannot pre-empt its pairs.

## Gate (T027/T028)

- **pytest** (Framework, evaluation, SimpleTrigger, research/outcomes, utils, NNNC): 643 passed.
  3 failed, the known pre-existing training-pipeline characterization tests, which have failed
  since before 023.
- **ruff** on the lines 024 added: clean. Three things are kept because they match their files'
  existing style: `Optional[...]` in `BaseStrategy`, the `REFUSED; return 1` idiom in
  `run_folds`, and a fall-back `try/except/pass` in `StopInvariants`.
- **Exit single-source test:** passes. `utils/FamilyExits.py` is classified as CONFIGURING, since
  it swaps values and holds no exit arithmetic.
- **Causality verdicts:** regenerated after the trigger-file edits. All 24 are identical to the
  pre-024 set (code sha `e2914f93881745b9`).
