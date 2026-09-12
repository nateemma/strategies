# Quickstart: labelfree-meta-labelling

Validation scenarios proving the feature works end to end. Run from the **repo root** (the
directory containing `user_data/`), **sequentially**.

**Prerequisites**: repo `.venv`; `regime/wf/pooled_nnnc.json` present with 378 trades / 15
folds; all 15 `saved_data/NNNC_WF_f*` model stores intact; feathers covering 2021-01-01 →
2026-09-07; nothing else running (`pgrep -f 'freqtrade backtesting'` empty).

```bash
export PYTHONPATH=user_data/strategies
PY=.venv/bin/python
ML=user_data/strategies/research/labelfree/metalabel
```

---

## S1 — The archive is what the plan assumes (FR-001)

```bash
$PY -c "
import json; d=json.load(open('user_data/strategies/regime/wf/pooled_nnnc.json'))
ev=[t for t in d['trades'] if int(t['fold_id'][1:])>=5]
print(len(d['trades']), d['folds_complete'], len(ev),
      round(100*sum(1 for t in ev if t['profit_abs']>0)/len(ev),1))"
```

**Expected**: `378 15 287 73.2` — the whole plan is calibrated to these four numbers. If any has
moved, the floor (4.681 pp/yr) and the base rate must be recomputed before anything else runs.

---

## S2 — Re-inference reproduces the archived entry set EXACTLY (FR-005b, SC-017)

```bash
$PY $ML/reinfer.py --archive regime/wf/pooled_nnnc.json \
                   --out research/labelfree/metalabel/rows.jsonl
```

**Expected**: exit 0, and a line per fold reporting `matches_exactly: true`.

**This is the gate.** A non-zero exit prints the symmetric difference and halts the feature. It
means the fold models, their pinned params, or the feather data changed since 2026-09-09 — which
is a finding, not an inconvenience. Do not proceed by dropping the unmatched trades.

---

## S2b — The probability columns exist, align, and cannot become features (R1)

```bash
$PY -c "
import sys; sys.path.insert(0,'user_data/strategies')
from utils.DataframeUtils import DataframeUtils
import pandas as pd, numpy as np
df = pd.DataFrame({'close':[1.,2.,3.], '%predict_prob_buy':[np.nan,0.9,0.4]})
print('dropped by remove_debug_columns:',
      '%predict_prob_buy' not in DataframeUtils().remove_debug_columns(df).columns)"
```

**Expected**: `True`. The `%` prefix routes the columns through `remove_debug_columns()`, which
is applied before scaling and in scaler creation — so the model's own output can never be
picked up as a training feature. Without the prefix this would be a
predict-your-own-prediction leak.

**Also check alignment.** `predictions` is front-padded when the model emits fewer rows than the
dataframe; the probability columns must use the **same offset** and be **NaN** outside it:

```bash
$PY -c "
import sys, json; sys.path.insert(0,'user_data/strategies')
rows=[json.loads(l) for l in open('research/labelfree/metalabel/rows.jsonl')]
import math
n_nan=sum(1 for r in rows if r.get('excluded_reason')=='nan_confidence')
print('rows excluded for NaN confidence:', n_nan, 'of', len(rows))"
```

**Expected**: a small count, reported rather than zero-filled. NaN marks bars the model never
scored (the warm-up region). Padding those with `0.0` would mark them "confidently not-buy" and
silently corrupt the earliest feature rows — which is why FR-006 excludes and counts them.

---

## S3 — The audit detects a leak it is required to catch (FR-004, SC-011)

```bash
$PY $ML/audit.py --rows research/labelfree/metalabel/rows.jsonl
```

**Expected**: every real feature passes causality-by-deletion, **and** the planted-leak control
is FLAGGED. An audit that passes its own planted leak certifies nothing and exits non-zero.

To confirm the check is not vacuous, verify the control is actually present:

```bash
$PY -c "
import sys; sys.path.insert(0,'user_data/strategies')
from research.labelfree.metalabel import audit
print('planted-leak control present:', hasattr(audit, 'PLANTED_LEAK'))"
```

**Why this scenario exists**: the first version of the analogous control in
`research/labelfree/liveness.py` passed vacuously because it did not actually sever anything.
The same failure one level up would certify a leaky feature table.

---

## S4 — No training trade closes after its evaluation fold opens (FR-007, R6)

```bash
$PY $ML/nested.py --rows research/labelfree/metalabel/rows.jsonl \
                  --out research/labelfree/metalabel/assignment.json
```

**Expected**: exit 0; `f05` reports `n_train: 91`; no `eval_fold` below `f05`.

---

## S5 — Every swept parameter is proven live before compute is spent (FR-012, SC-014)

```bash
$PY -c "
import sys; sys.path.insert(0,'user_data/strategies')
from research.labelfree.liveness import check_all
check_all(['veto_threshold','n_estimators','max_depth','min_child_samples'])"
```

**Expected**: every parameter `LIVE`. A `DEAD` parameter raises, and the sweep must not run.

**Why**: four parameters in this study were configured, logged, believed active, and reaching
nothing — 75+ runs void between them. Three were caught only by asserting on the value actually
applied.

---

## S6 — The veto trains nested and reports discrimination, not accuracy (SC-009, R3)

```bash
$PY $ML/model.py --rows research/labelfree/metalabel/rows.jsonl \
                 --assignment research/labelfree/metalabel/assignment.json \
                 --thresholds 0.4,0.5,0.6 --seeds 1,2,3,4,5 \
                 --out research/labelfree/metalabel/veto.json
```

**Expected**: AUC and precision at the operating point per threshold, as an interquartile mean
with a bootstrap interval over 5 seeds, with `n_train` beside every per-fold figure. Accuracy is
not printed.

**Kill check**: if AUC < 0.55 for every threshold, SC-009 fires — the feature is killed and the
report records that entry-time features carry no information about outcome beyond what the
primary already used.

---

## S7 — The P&L consequence, against the RECOMPUTED floor (SC-001, R2)

```bash
$PY $ML/evaluate.py --veto research/labelfree/metalabel/veto.json \
                    --archive regime/wf/pooled_nnnc.json --floor 4.681 \
                    --out research/labelfree/metalabel/result.json
```

**Expected**: a table whose every return row carries retained-trade count, trades/yr and mean
hold time, with the null configuration present as the control.

**The floor is 4.681 pp/yr, not 3.350.** The study-wide figure was measured on 378 trades over
3.68 years; this span is 287 over 2.45, so the floor is 40% higher. A delta between the two
numbers clears the old floor and is *not resolvable* here.

---

## S8 — Is the gain pair selection? (SC-010)

```bash
$PY $ML/attribution.py --result research/labelfree/metalabel/result.json \
                       --archive regime/wf/pooled_nnnc.json
```

**Expected**: all-pairs gain, ZEC-excluded gain, within-pair gain, and a verdict line.

`PAIR SELECTION` stops the feature and is recorded as such. Note the model **cannot** rank pairs
by construction (pair features are excluded, R7), so a large all-pairs / ex-ZEC gap here would
itself be surprising and worth diagnosing rather than reporting.

---

## S9 — The W3 persistence probe is labelled everywhere it appears (SC-018)

```bash
$PY $ML/model.py --probe W3 --out research/labelfree/metalabel/probe_w3.json
grep -c 'anti-causal' research/labelfree/metalabel/probe_w3.json
```

**Expected**: non-zero. The probe trains on folds *later* than 2023 and may never be quoted as
an out-of-sample return or counted toward SC-001.

---

## S10 — Trial log clean, and the report written even on a kill (SC-013, FR-025)

```bash
$PY -c "
import sys; sys.path.insert(0,'user_data/strategies')
from research.labelfree.trials import audit
ok,i = audit(); print('PASS', ok, i['n_pre_registered'], i['problems'])"
```

**Expected**: `PASS True <n> []`, with `<n>` grown by exactly the number of configurations run.

The protocol §6 report at `research/label-free-study/briefs/results/03-<date>.md` is written
**whether the feature is accepted or killed** — the directional finding is the deliverable.
