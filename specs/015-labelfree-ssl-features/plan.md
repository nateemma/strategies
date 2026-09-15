# Implementation Plan: labelfree-ssl-features

**Branch**: `015-labelfree-ssl-features` | **Date**: 2026-09-15 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `specs/015-labelfree-ssl-features/spec.md`

## Summary

Brief 07 asks whether a label-free learned embedding carries forward-return information the
hand-built indicators lack. It is designed to be **killed cheaply**: a Spearman screen gates
everything, and no backtest runs unless it passes.

Four layers: a TS2Vec encoder in MLX fitted per fold without labels; a three-way-split nested
ridge screen; a same-width random Gaussian control; and — only on a pass — an `NNNC_MLX` subclass
evaluated under protocol §5.

**One measurement taken before this plan changed its central gate.** The spec set the bar at a
constant ρ = 0.15 with the primary screen at horizon 48. Measured on our own fold panels, the
full 24-feature production set reaches **0.1594 at horizon 6** — reproducing the quoted ceiling
almost exactly — but only **0.0914 at horizon 48**. Pre-declaring 0.15 at horizon 48 asked the
embedding to beat a bar the *baseline itself cannot clear*. The bar is now the hand-built
features' own ρ, measured on the same data, pipeline and horizon.

## Technical Context

**Language/Version**: Python 3.12.13 under `freqtrade/.venv`

**Primary Dependencies**: **MLX 0.32.2** (GPU), numpy, pandas, scipy, scikit-learn (ridge).
**Nothing new is installed and nothing is downloaded** (FR-006, SC-016). `transformers` and every
pretrained model stay out of scope.

**Why MLX and not torch**: torch 2.13.0 is present but only transitively. The repository's NN
stack is MLX throughout — nine MLX GAN implementations, `MLXBaseClassifier`/`Predictor`/
`AnomalyDetector`, and `BaseNNStrategy` itself.

**Storage**:
- `research/labelfree/ssl/embeddings/<fold>_<input>_s<seed>.npz` — per-fold embeddings
- `research/labelfree/ssl/screen.json`, `pairagnostic.json`

**Testing**: pytest under `research/labelfree/tests/`, sequential.

**Project Type**: research library under `research/labelfree/`.

**Performance Goals**: the encoder trains on a fold's training range (~130K bars × 10 pairs). The
screen must complete in one session for all four input × horizon combinations.

**Constraints**:
- Sequential runs (FR-019); stage only this feature's files (FR-020).
- The outer freqtrade checkout receives no commits.
- **No backtest unless the primary screen passes** (FR-010).
- Every updated MLX array in the per-step `mx.eval` (FR-005).

**Scale/Scope**: the 15 anchored folds; the screen runs on a subset and reports per fold.

## Constitution Check

| Principle | Status | How this plan satisfies it |
|---|---|---|
| **I. Evaluate in the deployment regime** | PASS | The screen scores on fold **test** ranges; encoder and ridge are fitted on training ranges only. |
| **II. Persistence across distant eras** (NON-NEGOTIABLE) | **PASS, conditional** | Only reached if the screen passes; SC-006 then requires ≥10/15 folds and each of W1/W2/W3. If the screen fails, no candidate exists to persist and the feature ends at a KILL. |
| **III. Learnability is not edge** | **PASS, and it is the feature's core tension** | The screen is explicitly a *learnability* measure, and this repo has recorded that learnability is not edge — a +0.155 MCC once produced identical trades. So the screen is a **necessary-not-sufficient** filter: passing it earns a backtest, never an acceptance. Stated in the report. |
| **IV. Verify parameters are actually live** | PASS | The random control (FR-011) is exactly this check applied to feature width: it separates information from capacity. |
| **V. Lookahead-clean is not realistic** | **PASS, deferred** | The screen is a correlation, not a P&L claim, so costs do not apply to it. If a backtest follows, protocol §5 gates it under full costs (SC-005–SC-009). |
| **VI. Diagnose before tuning** | PASS | R1's measurement preceded and corrected the plan's central gate. |
| **VII. Validate cheaply; trackability before adaptivity** | **PASS — this brief IS the principle** | A one-session screen gates a week of backtests. |
| **VIII. One parameter space at a time** | PASS | The encoder's hyperparameters are fixed at TS2Vec defaults and not searched; searching them across four screens would be the multiple-testing failure R7 guards against. |
| **IX. Orthogonality earns a factor its place** | **PASS, and it is the hypothesis** | The embedding earns its place only by beating the hand-built features it would sit beside. |
| **X. Structural fixes, not per-pair overrides** | **PASS, and it is enforced** | FR-012a rejects an embedding whose scale depends on the pair, which is the per-pair-override failure in feature form. |

**Post-design re-evaluation**: unchanged. No deviations to record.

## Project Structure

### Documentation (this feature)

```
specs/015-labelfree-ssl-features/
├── spec.md
├── plan.md               # this file
├── research.md           # Phase 0 — 8 decisions; R1 corrects the spec's bar
├── data-model.md         # Phase 1
├── contracts/
│   └── ssl-cli.md        # Phase 1
├── quickstart.md         # Phase 1
├── checklists/requirements.md
└── tasks.md              # /speckit-tasks
```

### Source Code (repository root)

```
research/labelfree/
├── ssl/
│   ├── __init__.py
│   ├── ts2vec.py         # the encoder: dilated causal CNN + hierarchical contrastive loss, MLX
│   ├── inputs.py         # the two input types: engineered features, raw OHLCV windows
│   ├── screen.py         # three-way-split nested ridge; the hand-built bar measured alongside
│   ├── control.py        # same-width random Gaussian block through the identical pipeline
│   ├── pairagnostic.py   # FR-012a, at the pre-registered 0.5 ratio
│   ├── causality.py      # deletion test + planted leak
│   └── report.py         # the §6 result document
└── (reused unchanged) folds.py, features.py, labels.py, es/panel.py, stats.py
```

## Phases

| Phase | Content | Gate to pass before continuing |
|---|---|---|
| **0** | Research — the bar re-measured, the three-way split, MLX specifics | `research.md` written |
| **1** | Design — data model, CLI contract, quickstart | this document |
| **2** | Inputs + the hand-built bar measured per (input, horizon) | the bar reproduces R1's numbers |
| **3** | TS2Vec in MLX, with the causality deletion test and planted leak | a planted leak is detected |
| **4** | Pair-agnosticism check at the pre-registered threshold | reported; a rejection is a finding |
| **5** | **The screen** — four combinations, one primary | **BLOCKING**: a primary that does not beat its own bar is a KILL |
| **6** | Only on a pass: `NNNC_MLX` subclass + pooled walk-forward | protocol §5 |
| **7** | Result report | written whatever the outcome (FR-016) |

Phase 5 blocks 6. The expected path ends there, and the report is written either way.

## Complexity Tracking

No constitutional deviations.

The judgement worth recording is the opposite of complexity: **the brief is designed to be killed
for one session's work**, and R1's measurement made the gate coherent before any of it was spent.
A constant 0.15 at horizon 48 would have produced a confident KILL for a reason unrelated to the
hypothesis — the third instance of that shape in three briefs.
