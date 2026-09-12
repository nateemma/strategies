# Cross-Artefact Analysis Report — 011-labelfree-meta-labelling

**Date**: 2026-09-12 · **Stage**: pre-implementation (required before `/speckit-implement`)
**Artefacts**: spec.md, plan.md, research.md, data-model.md, contracts/metalabel-cli.md,
quickstart.md, tasks.md · **Mode**: read-only

## Specification Analysis Report

| ID | Category | Severity | Location(s) | Summary | Recommendation |
|----|----------|----------|-------------|---------|----------------|
| C1 | Coverage / Inconsistency | **CRITICAL** | tasks.md T034, `research/labelfree/stats.py:128` | **`activity()` cannot consume this feature's data.** T034 says to attach trades/yr and hold time "reusing `stats.py::activity`", but that function takes an **exposure path** `w` of shape `(n_pairs, T)` and derives trades from contiguous non-flat runs. This feature has **discrete archived trades** with explicit `open_date`/`close_date` — there is no exposure path. The reuse is a category error, and FR-014/SC-012 are mandatory on every reported return, so this blocks US3 entirely. | Compute activity directly from the trade list: `trades_per_year = retained / years`, `mean_hold_hours = mean(close_date - open_date)`. Do **not** synthesise an exposure path to feed `activity()` — that would invent a quantity. Add a distinct `trade_activity()` helper rather than overloading the existing one. |
| C2 | Coverage | **CRITICAL** | tasks.md T044, `research/labelfree/baselines.py:166` | **SC-003 is not satisfiable as tasked.** T044 says to run the four mandatory baselines "on the evaluation span", but `baselines.py` accepts `--fold` only (single fold, default `f08`) and has no aggregation across folds. SC-003 is an acceptance gate, so this is a blocking gap, not a convenience. | Add a `--folds` span mode to `baselines.py` that aggregates across folds 5–14 on the same annualisation the floor uses, or add an explicit task to build it. Either way T044 must not assume a capability that does not exist. |
| H1 | Underspecification | HIGH | tasks.md T046, spec.md SC-008, `stats.py:306` | **SC-008 has no natural mapping onto a nested design.** `overfitting_differential(screen_advantage, heldout_advantage, …)` compares a *screening* fold against a *held-out* fold. Here every evaluation fold is held out by construction and there is no screening fold, so "in-sample advantage" is undefined. Left as-is, T046 will be implemented by inventing a split, which is precisely the arbitrary choice the gate exists to prevent. | Define the split explicitly in the spec before implementing: the natural candidate is each fold-*k* model's advantage **on its own training trades** versus its advantage on fold *k*. Record the definition in research.md so it is pre-registered rather than chosen after seeing results. |
| H2 | Inconsistency | HIGH | quickstart.md S9, contracts/metalabel-cli.md §4, tasks.md T042 | **The W3 probe has no defined interface.** Quickstart S9 invokes `model.py --probe W3 --out …`, but the contract's `model.py` section defines only `--thresholds`, `--seeds`, `--dry-run`. T042 says to run the probe without naming a module or flag. The probe is SC-018, a mandatory reported artefact. | Add `--probe {W1,W2,W3}` to the `model.py` contract, specifying that it inverts the nested assignment and stamps `anti_causal: true` into its output. Update T042 to name the command. |
| M1 | Traceability | MEDIUM | tasks.md (9 tasks) | Nine FRs are **substantively covered but not cited**: FR-001 (T008/T019/T020), FR-002 (T014), FR-005a (T004/T009/T013), FR-007 (T022), FR-008 (T002/T022), FR-009 (T028), FR-019 (Phase 8 gate), FR-021 (Notes), FR-023 (T052). No work is missing; the audit trail is incomplete, which matters because FR-023 makes provenance an admissibility condition. | Add the FR ids to those task lines. No scope change. |
| M2 | Underspecification | MEDIUM | tasks.md T051, `research/labelfree/reproduce.py:31` | **"Extend `reproduce.py`" understates the work.** Its `CONFIG_KEYS` are `ArmConfig` fields and `rerun()` builds a subprocess that imports `research.labelfree.arms` and calls `train_arm`. A LightGBM meta-label configuration shares none of that machinery, so this is a second code path, not a parameter addition. | Either add a pluggable rerun backend to `reproduce.py`, or give the meta-label package its own `reproduce.py` and say so. Re-estimate T051 accordingly. |
| M3 | Constitution / Process | MEDIUM | tasks.md Phase 8 | **Gated tasks are prose-gated only.** "Do not start unless a configuration cleared SC-001 through SC-006" is a sentence, not a machine-readable state, and `/speckit-implement` walks unchecked boxes in order. Feature 010 hit exactly this: its gated Phase 6 tasks sat open and had to be explicitly cancelled at close. | Prefix T048–T050 titles with `[GATED]`, and add a Phase 9 task requiring they be marked cancelled-with-reason if the feature kills. |
| L1 | Scope drift | LOW | data-model.md TradeRow, spec.md Non-Goals | `target_profit` is described as "regression for sizing", while sizing is an explicit Non-Goal. Harmless if it stays unused, but it invites a sizing arm to appear mid-feature. | Keep the field, restate it as "retained for a future brief; unused here". |

## Coverage Summary

| Requirement group | Covered | Notes |
|---|---|---|
| FR-001…FR-006 (extraction, causality) | Yes | Phase 2 Gate C + Phase 3; 3 ids uncited (M1) |
| FR-005a/b/c (confidence, reproduction, no pair features) | Yes | Gates A and B |
| FR-007…FR-015 (model, evaluation) | Yes | Phases 4–5; SC-008 mapping undefined (H1) |
| FR-014 activity metrics | **Blocked** | **C1** — no usable implementation path |
| FR-016…FR-017 (attribution) | Yes | Phase 6 |
| FR-018…FR-020 (integration) | Yes | Phase 8, prose-gated (M3) |
| FR-021…FR-025 (process) | Yes | Phase 9; ids uncited (M1) |
| SC-001…SC-018 | 18/18 have tasks | SC-003 blocked (C2), SC-008 undefined (H1), SC-018 has no interface (H2) |

**Unmapped tasks**: none. Every task traces to at least one FR or SC.

## Constitution Alignment

No violations. The single declared deviation — W3 evaluated anti-causally (Principle II) — is
justified in plan.md Complexity Tracking, and the probe is barred from counting toward SC-001.
H2 is the one risk to it: a deviation that is justified but has **no interface to execute it**
would silently not happen, which is how a non-negotiable principle gets quietly dropped.

Worth noting positively: Principle IV is handled better here than in features 009/010 — T025/T026
require liveness probes *in their active regime* before compute is spent, which is the lesson
those features paid for.

## Metrics

- Functional requirements: **28** — covered 27, **blocked 1** (FR-014 via C1)
- Success criteria: **18** — tasked 18, of which 1 blocked (SC-003), 1 undefined (SC-008), 1 without an interface (SC-018)
- Tasks: **56** — format-valid 56/56, no duplicate ids, 8 parallel-marked
- Ambiguity count: **0** (no placeholders, no vague adjectives)
- Duplication count: **0**
- **Critical issues: 2**

## Next Actions

**Resolve C1 and C2 before `/speckit-implement`.** Both are blocking: C1 makes FR-014's
mandatory activity metrics unimplementable as written, and C2 makes SC-003 unsatisfiable. Both
are small fixes — a `trade_activity()` helper and a span mode for `baselines.py` — but both
would otherwise be discovered mid-implementation, after the gates had passed and compute had
been spent.

**Resolve H1 and H2 in the same pass.** H1 must be settled *before* results exist, or the
in-sample/held-out split becomes a post-hoc choice. H2 is the difference between the W3 probe
being run and being skipped.

M1, M2, M3 and L1 can follow or be accepted; none blocks.

Suggested sequence: edit tasks.md for C1/C2/H2/M1/M3, edit spec.md + research.md for H1, then
re-run `/speckit-analyze` until clean.

## Verdict

**NOT CLEAN — 2 CRITICAL, 2 HIGH.** Do not proceed to `/speckit-implement`. Every finding is an
artefact-level fix; none requires redesign, and the feature's structure — three blocking gates,
two kill points before integration — is sound.

---

# Re-run — 2026-09-12 (post-remediation)

All eight findings from the pass above were remediated. This is a fresh detection pass on the
amended artefacts, not a re-read of the previous report.

## Remediation applied

| ID | Was | Fix |
|----|-----|-----|
| **C1** | T034 reused `stats.py::activity`, which consumes an exposure path this feature does not have | New **T033a** implements `trade_activity()` from the trade list (`retained / years`, `mean(close − open)`), with an explicit "do NOT synthesise an exposure path" note; **T033b** tests it against a hand-checked fixture including a same-bar open/close |
| **C2** | `baselines.py` is single-fold; SC-003 is stated over the span | New **T043a** adds `--folds f05..f14` on the floor's own annualisation; **T043b** requires a one-fold span to reproduce `--fold` exactly; documented as contract §5b |
| **H1** | SC-008's in-sample/held-out split undefined under a nested design | **research R10** pre-registers it: fold-*k* model's advantage on its own training trades vs on fold *k*, aggregated across evaluation folds. SC-008 in spec.md now names R10. T046 cites it |
| **H2** | W3 probe invoked in quickstart with no contract interface | `--probe {W1,W2,W3}` added to the `model.py` contract with inversion semantics and the `anti_causal` stamp; new **T041a** builds it; T042 names the command |
| **M1** | 9 FRs covered but uncited | Ids added to T008, T009, T014, T022, T028, T052, Phase 8 header, T055 |
| **M2** | "Extend `reproduce.py`" understated a second code path | T051 rewritten; `metalabel/reproduce.py` added to the plan structure with the reason |
| **M3** | Phase 8 gated by prose only | T048–T050 prefixed `[GATED]`; new **T054a** requires they be marked cancelled-with-reason on a kill |
| **L1** | `target_profit` described as "for sizing", a Non-Goal | Restated as "retained for a future brief; unused here" |

## Fresh Detection Pass

| Pass | Result |
|---|---|
| A. Duplication | 0 — no near-duplicate requirements |
| B. Ambiguity | 0 placeholders; 0 vague adjectives (`robust`, `scalable`, `efficient`, `appropriate`, `reasonable` all absent) |
| C. Underspecification | 0 — H1 and M2, the two underspecified items, are now pinned |
| D. Constitution alignment | 0 violations; the one declared deviation now has an interface (H2) and cannot silently not happen |
| E. Coverage gaps | **0 of 28 FRs uncovered; 0 of 18 SCs uncovered** |
| F. Inconsistency | 0 — floor is 4.681 pp/yr wherever asserted (3.35 appears only contrastively); trade counts only 287/378; every module named in tasks is declared in the plan and vice versa; quickstart and contract agree on `--probe` |

## Metrics

- Functional requirements: **28** — covered **28**, blocked **0**
- Success criteria: **18** — covered **18**, blocked **0**, undefined **0**, without interface **0**
- Tasks: **62** (was 56; +6 remediation) — format-valid **62/62**, no duplicate ids, 10 parallel-marked, 3 `[GATED]`
- Research decisions: **10** (was 9; R10 added)
- Ambiguity: **0** · Duplication: **0**
- **Critical issues: 0** · High: **0** · Medium: **0** · Low: **0**

## Constitution Alignment

10/10 principles PASS. One declared deviation — W3 evaluated anti-causally (Principle II) —
justified in plan.md Complexity Tracking, now executable via `model.py --probe W3`, stamped
`anti_causal: true`, barred from SC-001, and required to be rendered with that flag wherever it
appears (SC-018).

## Verdict

**CLEAN.** No CRITICAL, HIGH, MEDIUM or LOW findings outstanding. The feature is ready for
`/speckit-implement`.

Two things worth carrying into implementation, neither a defect:

1. **T047 is expected to report `CANNOT BE EVALUATED`** for DSR/PBO — the trial log stores
   summary scalars, so seeds are the only axis. That is the correct outcome, not a failure to
   fix; satisfying SC-004 as written needs a study-wide trial-log schema change.
2. **Two kill points precede the expensive work** — SC-009 at T031 and SC-010 at T040, both
   before Phase 8's gated integration. A dead configuration never reaches strategy wiring.
