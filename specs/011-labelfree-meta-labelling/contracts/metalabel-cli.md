# Contract: metalabel CLI

Phase 1. The interfaces this feature exposes. Every command runs from the **repo root** (the
directory containing `user_data/`), **sequentially** — concurrent jobs put load 60 on 14 cores
and finished slower than one.

```bash
export PYTHONPATH=user_data/strategies
PY=.venv/bin/python
ML=user_data/strategies/research/labelfree/metalabel
```

---

## 1. `reinfer.py` — recover entry-time features from the saved primary models

```
$PY $ML/reinfer.py --archive regime/wf/pooled_nnnc.json \
                   --out research/labelfree/metalabel/rows.jsonl
```

| flag | meaning | default |
|---|---|---|
| `--archive` | pooled walk-forward record | required |
| `--folds` | comma-separated subset | all |
| `--out` | trade rows with feature vectors | required |

**Contract**

- Reloads each fold's saved primary model and re-runs inference over that fold's test range.
- Captures `pred_probs[:, BUY]` directly — no framework file is modified (R1).
- **Exits non-zero if the re-inferred `(pair, open_date)` entry set is not exactly the archived
  set** (FR-005b, SC-017), printing the symmetric difference. No partial output is written on
  failure.
- Emits one row per archived trade; unbuildable rows are excluded and **counted in the summary**.

**Refuses to run** if a fold's model file is missing — the feature depends on
`saved_data/NNNC_WF_f*` surviving, and a silent skip would train on a subset without saying so.

---

## 2. `audit.py` — prove no feature reads the future

```
$PY $ML/audit.py --rows research/labelfree/metalabel/rows.jsonl
```

| flag | meaning | default |
|---|---|---|
| `--rows` | trade rows to audit | required |
| `--skip-planted-leak` | **not available** | — |

**Contract**

- For each feature: recompute with all bars strictly after `open_date` deleted; assert the value
  is bit-identical (FR-003).
- Injects a **planted-leak control** — a feature equal to the trade's own `profit_abs` — and
  **fails if the audit does not flag it** (FR-004, SC-011). The control cannot be disabled: an
  audit that cannot detect a known leak certifies nothing.
- Runs the brief's permutation test alongside, reported separately. Permutation measures
  *informativeness*; deletion measures *causality*. They answer different questions and both are
  printed.
- Exits non-zero on any causality failure or on an undetected planted leak.

---

## 3. `nested.py` — build the nested assignment

```
$PY $ML/nested.py --rows <rows> --out research/labelfree/metalabel/assignment.json
```

**Contract**

- Fold *k*'s training set contains only trades whose **`close_date`** precedes fold *k*'s
  `test_start` (R6).
- Folds 0–4 are warm-up and never appear as `eval_fold`.
- Emits `n_train` per fold; `f05` is expected to be 91.
- Exits non-zero if any training trade's `close_date` is not strictly before its eval fold.

---

## 4. `model.py` — train and evaluate the veto

```
$PY $ML/model.py --rows <rows> --assignment <assignment> \
                 --thresholds 0.4,0.5,0.6 --seeds 1,2,3,4,5 \
                 --out research/labelfree/metalabel/veto.json
```

| flag | meaning | default |
|---|---|---|
| `--thresholds` | veto operating points | `0.4,0.5,0.6` |
| `--seeds` | ≥5 required | `1,2,3,4,5` |
| `--dry-run` | pre-register only, do not train | off |

**Contract**

- Appends every configuration to `research/labelfree/trials.jsonl` **before** training it
  (FR-011), including its pinned capacity parameters.
- **Refuses to run until every swept parameter has been proven live** (FR-012, SC-014), via the
  existing `research/labelfree/liveness.py` registry extended to this model.
- Reports AUC **and** precision at the operating point; **never accuracy** (R3) — a
  majority-class model scores 73.2% and that number cannot distinguish skill.
- Reports the interquartile mean with a bootstrap interval across seeds, never a best seed.
- Emits `n_train` beside every per-fold figure.

---

## 5. `evaluate.py` — the P&L consequence

```
$PY $ML/evaluate.py --veto <veto.json> --archive regime/wf/pooled_nnnc.json \
                    --floor 4.681 --out research/labelfree/metalabel/result.json
```

**Contract**

- Replays the archived trades with vetoed trades removed and compares to the unvetoed archive
  over the identical span — the **control is the unvetoed primary**, i.e. exactly what the veto
  would replace.
- Quotes the **recomputed 4.681 pp/yr** floor (R2), never the study-wide 3.350.
- Every return carries retained-trade count, trades/yr and mean holding time (FR-014, SC-012).
- A configuration retaining too few trades to resolve is emitted as `inconclusive: true` **with
  its return withheld** (FR-015).
- The null configuration (accept everything) always appears as the control row.

---

## 6. `attribution.py` — is it pair selection?

```
$PY $ML/attribution.py --result <result.json> --archive regime/wf/pooled_nnnc.json
```

**Contract**

- Recomputes the gain with ZEC excluded (SC-010) and reports it beside the all-pairs figure.
- Decomposes the gain **within** pairs and states whether discrimination survives there.
- Prints a verdict line: `PAIR SELECTION` or `WITHIN-PAIR`. The former stops the feature.

---

## 7. `NNNC_Meta.py` — strategy integration

**Written only if a configuration is accepted** (FR-019, SC-016). A thin `NNNC_MLX` subclass:

- loads the primary model from a **pinned** path and does not retrain it (FR-018);
- consults the secondary model at entry and declines the trade below the threshold;
- merges parameter dictionaries from the parent rather than redefining them (constitution);
- any divergence between its trades and the offline replay is **reported**, not averaged away
  (FR-020).

---

## Exit codes (all commands)

| code | meaning |
|---|---|
| 0 | success |
| 1 | a gate failed — causality, planted leak, reproduction, or liveness |
| 2 | inputs missing (archive, saved model, rows) |

A non-zero exit from `reinfer.py` or `audit.py` **halts the feature**. Nothing downstream is
evidence until they pass.
