# Implementation Plan: labelfree-meta-labelling

**Branch**: `011-labelfree-meta-labelling` | **Date**: 2026-09-12 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/011-labelfree-meta-labelling/spec.md`

## Summary

Train a secondary model on the realised outcomes of trades the primary strategy actually took,
using only information available at each trade's entry bar, and measure whether vetoing its
low-confidence trades raises net return. The primary model is not touched.

The training set already exists and is out-of-sample by construction: 378 archived walk-forward
trades, of which folds 5–14 (287 trades, 2.45 yr) are the evaluation span. The work is four
staged pieces — build the trade table and prove it cannot see the future; train the nested veto;
measure the P&L consequence; determine whether the gain is pair selection — with strategy
integration gated on acceptance.

## Technical Context

**Language/Version**: Python 3.12.13 (repo `.venv`)
**Primary Dependencies**: LightGBM 4.7.0 (secondary model), scikit-learn 1.9.0 (metrics, CV
utilities), NumPy/pandas, MLX (re-inference of the saved primary models), xgboost 3.2.0
(available as a cross-check only)
**Storage**: JSON/JSONL artefacts under `research/labelfree/metalabel/`; the trial log is the
existing `research/labelfree/trials.jsonl`
**Testing**: pytest under `research/labelfree/tests/`
**Target Platform**: macOS / Apple Silicon, local
**Project Type**: single research module inside the strategies repository
**Performance Goals**: one re-inference pass per fold (15 total); secondary-model training is
seconds per configuration at n ≤ 287
**Constraints**: all runs sequential; no feature may read data after a trade's `open_date`;
re-inference must reproduce the archived entry set exactly; the primary model, its guards, its
exits and its stop are untouched. The only `Framework/` change is three `%`-prefixed debug
columns (R1), which are dropped before scaling and cannot become features
**Scale/Scope**: 378 archived trades, 15 folds, 10 evaluation folds, ~9 features, 5 seeds per
configuration

**Measured inputs this plan depends on** (2026-09-12):

| quantity | value | source |
|---|---|---|
| pooled archive | 378 trades / 15 folds | `regime/wf/pooled_nnnc.json` |
| evaluation span (folds 5–14) | 287 trades, 2.45 yr | same |
| **recomputed floor** | **4.681 pp/yr** | research R2 |
| base win rate | 73.2% | same |
| per-fold primary models | 15 present, ~51 KB safetensors | `saved_data/NNNC_WF_f*` |

## Constitution Check

| Principle | Status | How this plan satisfies it |
|---|---|---|
| I. Evaluate in the deployment regime | **PASS** | Evaluation is folds 5–14, the most recent 2.45 yr. Warm-up folds are excluded from every claim. |
| II. Persistence across distant eras (NON-NEGOTIABLE) | **PASS, with a declared deviation** | W1 and W2 are evaluated out-of-sample. W3 (2023) has no nested training data and is run as an explicitly anti-causal probe, never quoted as a return. Recorded in Complexity Tracking. |
| III. Learnability is not edge | **PASS** | AUC is a screen (SC-009) and explicitly not the verdict. SC-001 is P&L against the recomputed floor. US3 exists because discrimination without a P&L consequence is not a result. |
| IV. Verify parameters are actually live | **PASS** | FR-012/SC-014 require every swept parameter proven live before its sweep. `research/labelfree/liveness.py` is extended to cover the secondary model's parameters. |
| V. Lookahead-clean is not realistic | **PASS** | The target is realised `profit_abs` under real exits, fees and fills — execution constraints are already baked into the archive rather than modelled. FR-002/FR-003 add the causality guarantee on the feature side. |
| VI. Diagnose before tuning | **PASS** | Phase order is extract → audit → train → measure. The leakage audit runs before any model is trained. |
| VII. Validate cheaply first | **PASS** | US1 (the audited trade table) is a standalone deliverable reusable by briefs 5 and 8 regardless of whether any veto works. |
| VIII. One parameter space at a time | **PASS** | Veto before sizing, as the brief directs. Sizing is out of scope for this feature. |
| IX. Orthogonality earns a factor its place | **PASS** | The veto is measured against the unvetoed primary, i.e. against exactly what it would replace. |
| X. Structural fixes, not per-pair overrides | **PASS** | Pair features are excluded by design (R7), so the model cannot express a per-pair override. |

**Repository and code constraints**: commits go only to `user_data/strategies`; staging is
file-by-file; `Framework/` is not modified (R1 resolves the confidence capture feature-locally);
no parameter dictionary is redefined wholesale.

**Experiment workflow gates**: pinned time ranges (fold plan), distant-era window (R6 probe),
realistic execution (inherited from the archive), liveness proof (FR-012), paired seed
robustness (FR-010, ≥5 seeds, IQM), trade count reported (FR-014), scalers — not applicable, the
primary's feature list is unchanged.

## Project Structure

### Documentation (this feature)

```
specs/011-labelfree-meta-labelling/
├── spec.md
├── plan.md              # this file
├── research.md          # Phase 0
├── data-model.md        # Phase 1
├── quickstart.md        # Phase 1
├── contracts/
│   └── metalabel-cli.md # Phase 1
├── checklists/
│   └── requirements.md
└── analyze.md           # written by /speckit-analyze
```

### Source Code (repository root = `user_data/strategies`)

```
Framework/
└── BaseNNStrategy.py # +3 debug columns in get_predictions(): %predict_prob_{sell,hold,buy}
                      #  NaN-initialised at full length, written at the SAME offset as
                      #  `predictions`, AFTER the (currently inert) Markov branch.
                      #  Strategy-agnostic; dropped by remove_debug_columns(). (R1)

research/labelfree/metalabel/
├── __init__.py
├── archive.py        # read the pooled WF archive -> trade rows
├── reinfer.py        # replay saved fold models, READ the %predict_prob_* columns (R1, R8)
├── features.py       # assemble the entry-time feature vector (pair features EXCLUDED, R7)
├── audit.py          # causality-by-deletion + planted-leak control (R5)
├── nested.py         # fold-k trains on trades closed before fold-k test_start (R6)
├── model.py          # LightGBM veto, capacity pinned for n=91 (R4)
├── evaluate.py       # veto replay, P&L consequence, activity metrics
├── attribution.py    # ZEC-excluded and within-pair decomposition (SC-010)
├── reproduce.py      # fresh-process reproduction; a SECOND path, not an extension of
                      #  research/labelfree/reproduce.py, whose CONFIG_KEYS and subprocess
                      #  are ArmConfig/train_arm specific (analyze M2)
└── report.py         # protocol section 6 assembly, incl. the "not resolved" rendering (SC-007)

research/labelfree/tests/
└── test_metalabel.py

NNNC/
└── NNNC_Meta.py      # WRITTEN ONLY ON ACCEPTANCE (FR-019)
```

**Structure decision**: a package under the existing `research/labelfree/`, reusing that
module's `trials.py`, `stats.py` and `costs.py` rather than duplicating them. Duplicated
statistics drift from the code that runs — a failure this study has already recorded, and the
same reasoning is why the primary's probabilities are *stored by the framework* rather than
recomputed by this feature (R1). The `Framework/` change is three debug columns and nothing
else.

## Phases

- **Phase 0 — Research** *(complete)*: `research.md`. Ten decisions, including the recomputed
  floor (4.681 pp/yr), the discovery that the primary's confidence is computed and discarded
  (R1), and the pre-registered in-sample/held-out split for SC-008 under a nested design (R10).
- **Phase 1 — Design** *(complete)*: `data-model.md`, `contracts/metalabel-cli.md`,
  `quickstart.md`.
- **Phase 2 — Tasks**: `/speckit-tasks`. First tasks are the leakage audit and the re-inference
  reproduction gate; last are the protocol §6 report and the trial-log audit.
- **Phase 3 — Implement**: `/speckit-implement`, sequential runs only, after a clean
  `/speckit-analyze`.

## Complexity Tracking

| Deviation | Why needed | Simpler alternative rejected because |
|---|---|---|
| **W3 evaluated anti-causally** (Principle II) | W3 lies entirely inside the warm-up folds; no nested model can precede it. Without the probe the feature ships with W1 and W2 only, which are four months apart — no distant-era evidence at all, which is what Principle II exists to prevent. | Dropping W3 (leaves the non-negotiable principle unsatisfied); substituting three thirds of the evaluation span (adjacent windows share a regime — this repo's own recorded finding, so it is a weaker test that looks like a stronger one). Mitigation: the probe is labelled at every appearance, may never be quoted as an OOS return, and never counts toward SC-001. |
| **Re-inference of 15 primary models** | The archive does not store the primary's entry confidence, and "when to trust the primary" is untestable without it. | Using OHLCV-only features (tests a weaker hypothesis than the brief states); joining features to archived trades without a reproduction check (the confidence might not be the value that triggered the entry). Mitigation: exact entry-set reproduction gate (R8, FR-005b). |
| **Three `%`-prefixed debug columns added to `Framework/BaseNNStrategy`** | The primary's probabilities are computed, smoothed, printed, and then discarded. Storing them is a two-line change at a site already handling them; the alternative duplicates the prediction path. | Calling the classifier directly from `reinfer.py` (rejected: a duplicated prediction path drifts from the code that runs — the failure this study already recorded); stashing `pred_probs` on `self` (rejected: hidden cross-call state). Mitigation: the `%` prefix routes them through `remove_debug_columns()`, so they can never enter the feature list; the change is strategy-agnostic (probabilities every NN strategy already computes) and adds nothing strategy-specific. |
