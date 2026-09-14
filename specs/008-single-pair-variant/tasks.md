---
description: "Task list for feature 008 — Single-Pair (LINK) Variant of NNNC_MLX"
---

# Tasks: Single-Pair (LINK) Variant of NNNC_MLX

**Input**: spec.md (21 FRs, 8 SCs, 4 user stories), plan.md, research.md (R1-R10),
data-model.md, contracts/analysis-cli.md, quickstart.md (S1-S6)

## Format: `[ID] [P?] [Story] Description`

All execution is sequential (FR-017). `[P]` marks authoring-only parallelism.

---

## Standing conventions — binding on EVERY task

Cited once here; T018 verifies each individually. Coverage of exactly this block was asserted
prematurely in four prior features of this project.

- **FR-017**: sequential only. Kill with `pkill -9 -f 'freqtrade backtesting'`, never bare
  `freqtrade` — it matches the live dry run.
- **FR-018**: `Framework/`, `run_arm.py`, `compare.py`, `regimedata.py`, `evaluation/` are NOT to be
  modified.
- **FR-019 / SC-008**: no gate, exit, label or architecture parameter differs between arms. The
  pair universe is the only difference.
- **FR-004**: the ungated-gate warning appearing in a run VOIDS that run.
- **FR-005**: verify the model artefact exists; do not trust exit status alone.
- **FR-011**: zero trades ⇒ zero-trade, never zero return.
- **FR-015**: no classifier metric feeds a verdict.
- **FR-020**: an incomplete fold set is reported PARTIAL.
- **FR-021**: win rate is NOT reported as quality below n=20.
- Exit tags carry a TRAILING SPACE — always `.strip()` an `exit_reason`.

---

## Phase 1: Setup (no compute)

- [X] T001 Create `config/config_link.json` from `config/config.json` with the whitelist reduced to
      exactly `["LINK/USDT"]`, preserving all other settings (FR-001)
- [X] T002 Create `NNNC/NNNC_MLX_LINK.py`: `class NNNC_MLX_LINK(NNNC_MLX): pass`. The distinct
      class name is what gives it its own model directory (FR-002; **R4**)
- [X] T003 Create `NNNC/NNNC_MLX_LINK.json` as a **FULL** copy of `NNNC_MLX.json` — buy AND sell
      AND roi/stoploss/trailing — with `strategy_name` rewritten (FR-003; **R4** — a buy-only file
      silently reverts the exit stack, measured at -2.4pp on 2026-09-09)
- [X] T004 Verify isolation per quickstart S1: whitelist is exactly LINK, `strategy_name` matches
      the class, and the params file carries a `sell` block (SC-001)

## Phase 2: US1 — the model exists and is honestly configured (P1)

- [X] T005 [US1] Train `NNNC_MLX_LINK` on 20220101-20250831 with `-c config/config_link.json`
      (~12 min, seed 1)
- [X] T006 [US1] Verify `saved_data/NNNC_MLX_LINK/*.safetensors` exists (FR-005) **and** that
      `COMPLETELY UNGATED` does not appear in the training log (FR-004). Either failure stops the
      feature here — do not proceed to measurement with a void model
- [X] T007 [US1] Confirm the model directory is distinct from `saved_data/NNNC_MLX/` and that the
      control's artefact is untouched (FR-002)

**Checkpoint**: a LINK-only model exists, gated identically to the control.

## Phase 3: Evaluation (no verdict yet)

- [X] T008 Run `run_arm.py --strategy NNNC_MLX_LINK --window W1|W2|W3 --tag link_only
      --config config/config_link.json`, sequentially (quickstart S3)
- [X] T009 Confirm each arm's `returncode == 0` and record trade counts. **~8 trades on W1 is the
      expected result, not a failure** (FR-011)

## Phase 4: US2 — which LINK trades each model takes (P1) 🎯 THE FEATURE'S ANSWER

- [X] T010 [US2] Implement `regime/pair_restricted.py` per contracts/analysis-cli.md. It MUST
      output all four groups, per window: (1) selection — shared / identical / variant-only /
      control-only with each differing trade's outcome; (2) per-trade expectancy with a bootstrap
      interval per arm; (3) total-return delta with CI and the applicable noise floor, marked
      UNRESOLVED when smaller; (4) guards — zero-trade recorded as such, win rate suppressed below
      n=20, `exit_reason` stripped (FR-006; **R7** — no existing tool does this)
- [X] T011 [US2] Feed `compare.py`'s bootstrap via its trade-list entry point rather than
      reimplementing the statistic (FR-018; **R7**)
- [X] T012 [US2] Report the selection comparison for W1, W2 and W3. **Identical trade sets are a
      POSITIVE finding** — cross-pair training contributes nothing to LINK selection — and must be
      reported as such, not as a null (FR-010, SC-002)

## Phase 5: US3 — per-trade expectancy on LINK (P1)

- [X] T013 [US3] Report per-trade expectancy with a bootstrap interval for both arms, per window
      (FR-007, SC-003). **R3**: at W1's n=8 and sd 1.47 the SE is ~0.52pp, so expectancy IS
      resolvable where total return is not
- [X] T014 [US3] Report every total-return delta with its CI and the applicable noise floor, marking
      anything smaller than the floor **UNRESOLVED** (FR-008, FR-009, SC-004)
- [X] T015 [US3] Suppress win rate on any window with fewer than 20 trades (FR-021 — W1's 8 LINK
      trades are 100% winners; one added loser moves it 11 points on no evidence)

## Phase 6: US1 — seed robustness (P2)

- [ ] T016 [US1] *(DEFERRED — see T019: with 38 of 39 trades identical between two models trained on entirely different data, seed variance cannot be the interesting question)* Retrain with `MLX_SEED=7` under a seed-namespaced class so seed 1's weights are not
      overwritten, re-run the arms, and report identical-across-seeds as ONE measurement (FR-016)

## Phase 7: US4 — pooled walk-forward (P3, CONDITIONAL)

> **GATE**: run this ONLY if T012 shows the trade sets DIFFER. If the two models select identical
> LINK trades, pooled folds cannot add information and this phase is skipped and recorded as
> skipped. ~1.5 h.

- [X] T017 [US4] **SKIPPED — gate did not open (trade sets are identical).** If and only if the gate opens: build a fold plan excluding any fold whose TEST
      range predates LINK's 2022-01-14 listing (FR-013; **R9**), run it, and report pooled trade
      count and achieved floor with the delta (FR-014, SC-006)

## Phase 8: Verification and conclusion

- [X] T018 **Verify the standing-conventions block, one named check each**: FR-004 (warning absent
      from every run used as evidence), FR-005 (artefact exists), FR-011 (zero-trade recorded as
      such), FR-015 (no classifier metric in any verdict), FR-017 (no overlapping runs), FR-018
      (`git status` clean on Framework/, run_arm.py, compare.py, regimedata.py, evaluation/),
      FR-019/SC-008 (no parameter differs), FR-020 (if a fold set ran and was
      incomplete it was marked PARTIAL; if US4 was skipped, record N/A), FR-021 (win rate suppressed)
- [X] T019 Write `regime/SINGLE_PAIR_LINK.md`: the selection comparison, the expectancy comparison
      with intervals, every delta with its floor, **the pair-dominance rotation (FR-012, SC-005)**,
      and a plain-language answer to whether cross-pair training helps LINK selection, harms it, or
      cannot be distinguished — stating the required sample size if the last (SC-007)
- [X] T020 Record the outcome in memory, including the rotation table, which is a finding
      independent of whatever the variant shows
- [X] T021 Commit: stage only this feature's files. **Never `git add -A`**

---

## Dependencies

- T001-T004 → T005 → T006 (**hard stop on failure**) → T008 → T010-T015 → T018-T021
- T016 after T012; T017 gated on T012's outcome
- T010 before T012; T012 before T017's gate decision

## Implementation Strategy

**MVP = Phases 1-5** (T001-T015, ~30 min): the model plus the two questions that its trade count
can actually answer. That is the whole feature; Phase 7 is a conditional extra and Phase 6 is
robustness.

**Stop conditions**: if T006 finds the ungated warning or a missing artefact, stop — every number
after it is void. If T012 shows identical trade sets, skip Phase 7 and go straight to the
conclusion; that IS the answer.

**Do not** declare a winner on a single-window P&L delta. At 8 W1 trades the instrument cannot
support it, and that is the specific error this feature is shaped to prevent.
