# Implementation Plan: labelfree-generative-stress

**Branch**: `014-labelfree-generative-stress` | **Date**: 2026-09-14 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `specs/014-labelfree-generative-stress/spec.md`

## Summary

Brief 06 builds the robustness veto every other brief assumes exists: *how does a policy behave on
plausible paths it has never seen?*

The feature is small, because two measurements made it small.

1. **None of the three rebuilt generators can emit a coherent sequence** — TabDDPM and WGAN
   return `(n, 1, F)`, CTAB-GAN+ returns a DataFrame. They are class-conditional row samplers,
   and a row sampler cannot produce a price *path* for a path-dependent policy.
2. **Clarification Q3 removed CGAN too**, the only sequence-capable type, as unmaintained.

So there is exactly one generator: a **stationary block bootstrap** over real per-pair panels,
resampling prices, indicators, cost *and* the model's own class probabilities together. The
policy is scored by feature 013's exit surrogate, reused unchanged at 31 ms per evaluation.

Four layers: the bootstrap, its calibration against the incumbent's real pooled-WF P&L, a
persisted path bank with provenance, and a distributional stress report.

## Technical Context

**Language/Version**: Python 3.12.13 under `freqtrade/.venv`

**Primary Dependencies**: numpy, pandas, scipy (`scipy.stats.ks_2samp`). **No GAN dependency** —
nothing in `GANs/` is imported, per clarification Q3.

**Storage**:
- `research/labelfree/stress/banks/<fold-set>_L<block>_s<seed>.npz` — path banks
- `research/labelfree/stress/calibration.json`, `stress_report.json`

**Testing**: pytest under `research/labelfree/tests/`, sequential.

**Project Type**: research library under `research/labelfree/`.

**Performance Goals**: ≥1,000 paths × ~90-day panels. At the surrogate's measured 31 ms this is
≈30 s per fold-set sweep; sampling the bank is the one-off cost and is done offline.

**Constraints**:
- Every run sequential (FR-018); stage only files this feature touches (FR-019).
- The outer freqtrade checkout receives no commits.
- No policy is trained on synthetic data (FR-017).
- Sampling happens offline into a persisted bank, never inside a loop (FR-001).

**Scale/Scope**: the 15 anchored folds' test panels already on disk from feature 013.

## Constitution Check

| Principle | Status | How this plan satisfies it |
|---|---|---|
| **I. Evaluate in the deployment regime** | PASS | Blocks are drawn from the fold **test** ranges — the deployment regime — not from the anchored training spans. |
| **II. Persistence across distant eras** (NON-NEGOTIABLE) | **N/A, recorded** | This feature ships no policy and makes no acceptance claim. It is a veto instrument; SC-009 forbids promoting anything on its output. Principle II binds the policies it vetoes, which is clarification Q1's answer. |
| **III. Learnability is not edge** | PASS | No ML metric appears anywhere. Calibration is on the P&L distribution itself. |
| **IV. Verify parameters are actually live** | PASS | The block length `L` is **swept**, not chosen, and reported per value — the same discipline feature 013's liveness probe applied to its search space. |
| **V. Lookahead-clean is not realistic** | PASS | Cost and slippage ride along inside each resampled block, since they are panel columns. |
| **VI. Diagnose before tuning** | PASS | The generator census (R7) preceded the design and is what shaped it. |
| **VII. Validate cheaply; trackability before adaptivity** | PASS | The whole feature is the cheap validation other briefs assume. The sequence-capability probe cost minutes and removed three quarters of the planned work. |
| **VIII. One parameter space at a time** | PASS | One parameter (`L`), swept alone. |
| **IX. Orthogonality earns a factor its place** | PASS | No factors. |
| **X. Structural fixes, not per-pair overrides** | PASS | One bootstrap procedure across all pairs. |

**Post-design re-evaluation**: unchanged. No deviations to record in Complexity Tracking.

**A note on Principle II being N/A.** Marking a NON-NEGOTIABLE principle inapplicable deserves
scrutiny, since feature 013's analyze pass caught that principle *claimed but unimplemented*. The
difference is that 013 shipped a candidate operating point and had to demonstrate persistence;
this feature ships a measuring instrument and explicitly forbids promotion on its output (SC-009,
FR-014). If that ever changes — if a stress result is used to *support* rather than veto — this
row becomes a violation.

## Project Structure

### Documentation (this feature)

```
specs/014-labelfree-generative-stress/
├── spec.md
├── plan.md               # this file
├── research.md           # Phase 0 — 8 decisions, incl. the honest SC-001 weakness
├── data-model.md         # Phase 1
├── contracts/
│   └── stress-cli.md     # Phase 1 — command surface
├── quickstart.md         # Phase 1
├── checklists/requirements.md
└── tasks.md              # /speckit-tasks
```

### Source Code (repository root)

```
research/labelfree/
├── stress/
│   ├── __init__.py
│   ├── bootstrap.py      # stationary block bootstrap over panels (Politis & Romano)
│   ├── bank.py           # persist/load path banks with provenance
│   ├── calibrate.py      # the Tail-GAN criterion, at PRE-REGISTERED tolerances
│   ├── gencensus.py      # sequence capability of each generator, from running code
│   ├── run.py            # stress a policy across a bank -> distribution
│   └── report.py         # the §6 result document
├── es/                   # REUSED unchanged — exits.py, evaluate.py, panel.py
└── stats.py              # REUSED — bootstrap_ci

research/labelfree/tests/
├── test_stress_bootstrap.py
├── test_stress_bank.py
└── test_stress_calibrate.py
```

**Reuse is deliberate and is the point.** `es/panel.py` already assembles the exact per-pair
panel a path needs; `es/exits.py` already scores a panel with measured fidelity. This feature
writes a resampler, a calibration and a report — nothing else.

## Phases

| Phase | Content | Gate to pass before continuing |
|---|---|---|
| **0** | Research — the generator census, the path definition, pre-registered tolerances | `research.md` written |
| **1** | Design — data model, CLI contract, quickstart | this document |
| **2** | Generator census (US3) | sequence capability recorded for every in-scope type |
| **3** | Stationary bootstrap + bank, with tests | a bank and a fold panel score identically on the same bars |
| **4** | **Calibration** at the pre-registered tolerances (US1) | **BLOCKING**: if the bootstrap fails, the feature is killed (SC-007) — there is no fallback generator |
| **5** | Stress run over ≥1,000 paths (US4) | distribution reported, never a point estimate |
| **6** | Result report, §6 template | written whatever the outcome (FR-021) |

Phase 4 blocks 5–6. Clarification Q3 left no second generator, so a calibration failure is a kill
rather than a fallback — which is exactly why its tolerances are fixed in research.md R3 before
any run.

## Complexity Tracking

No constitutional deviations. The one judgement worth recording is the opposite of complexity:
**three quarters of the brief's planned work was removed by a measurement that cost minutes** —
the generator census showed the GAN adapters could not be built as specified, and clarification
Q3 removed the remaining candidate. What survives is the block bootstrap the brief required as a
baseline regardless.
