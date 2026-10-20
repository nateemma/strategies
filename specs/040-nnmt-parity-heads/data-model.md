# Data Model: 040 NNMT Parity and Head Study

## Arm

| Field | Meaning |
|---|---|
| `tag` | One of: `mt`, `mt_fw0`, `mt_op`, `mt_raw`, `mt_lr`, `mt_lr_raw`, `mt_lr_ri`, `<M>_fw0`, `<M>_g24`, `<M>_hpa`, `<M>_gate`, `<M>_tp`, `<M>_tr`, `<M>_cs`, `<M>_ah`, head arms `hd_<head>`, transition arms `tr_gate`, `tr_scale`, `tr_long`, `tr_exit`, `<M>_trh` |
| `class` | the strategy class (run_folds generates fold classes `<class>_fNN`, pinned with `MODEL_TRAIN_TIMERANGE` and `MODEL_SEED`) |
| `plan` | `evaluation/folds/plan_<tag>.json` (family `NNMT` unless the arm runs on an NNNC book; 15 folds, config_wf, seed 1) |
| `comparator` | the arm it is judged against (§ Stage comparators) |
| `stage` | models / framework / sizer / heads / transition |

`<M>` = the R7-selected model arm.

**Fold record**: `regime/arms/<tag>_fNN_FNN.json`. Holds the trades, `trades_hash`, `n_trades` and
`model_key`.

## Stage comparators

| Stage | Arm | Comparator | Decision |
|---|---|---|---|
| models | `mt` | — | reported beside `wf_aro_g24`, `cs_aro`, `w_aro`, `st_g24_rev` |
| models | `mt_fw0` | `mt` | identity: `trades_hash` equal on 15/15 folds |
| models | `mt_op` | `mt_fw0` | `judge_one` (reported; the operating point is the stage default if model-decided share ≥ 0.5) |
| models | `mt_raw`, `mt_lr`, `mt_lr_raw`, `mt_lr_ri` | `mt_op` | `judge_one`; R7 selects `<M>` |
| framework | `<M>_g24`, `_hpa`, `_gate`, `_tp` | `<M>_fw0` (= `<M>`) | `judge_one`; PASS → candidate default |
| framework | `<M>_tr` (trading head only) | `<M>` | `judge_one` (does multi-task help?); `wf_aro_g24` reported |
| sizer | screen | R9 bar | pass rule |
| sizer | `<M>_cs` | `w_aro` (must PASS); `cs_aro` (reported) | `judge_one` |
| heads | `<M>_ah` trading head | `<M>` | reported (side check) |
| heads | `hd_<head>` | its unmodified book (`cs_aro` or `w_aro`, per R10) | `judge_one` |
| transition | `tr_gate`, `tr_scale`, `tr_exit` | `cs_aro` | `judge_one` |
| transition | `tr_long` | sign test vs 0 per half; `st_g24_rev` reported | pre-registered rule |
| transition | `<M>_trh` | the stand-in cell's screen | AUC within 0.01 both halves → its tables replace the stand-in's |

## Model-decided share (per arm, per fold)

| Field | Meaning |
|---|---|
| `guard_pass_bars` | bars passing the guard chain (from the signal table's KEEP columns) |
| `entries` | bars with `enter_long` |
| `model_share` | 1 − entries ÷ guard_pass_bars (higher = the model decides more) |

## Head-probability table (`research/nnnc_framework/signals/<M>_fNN.parquet`)

The 039 columns (date, pair, p_sell/p_hold/p_buy, enter_long, enter_tag, KEEP), plus
`p_<task>_<class>` for regime (bear/side/bull), risk (low/normal/high), momentum and flow
(down/neutral/up), and profit (loss/neutral/profit).

## Head screen result (`research/nnmt/screens_heads.json`)

Per head:
- `label_auc`: early, late;
- `baseline_auc`: early, late (the causal trailing state);
- `margin` = label_auc − baseline_auc, for each half;
- `skill_pass` (margin ≥ 0.02 in both halves);
- `trades_gap`, `partial`, `sign_folds`, `relevance_pass` (R9 metric, early only, in the declared
  direction);
- `pass` = skill_pass AND relevance_pass;
- `rho_pbuy` (orthogonality, reported).

## Regime-transition target

| Field | Meaning |
|---|---|
| `H` | horizon in bars: 96, 288 or 672 (L = H) |
| `level` | `pair` or `btc` |
| `state_t` | S_H(t): BULL/SIDEWAYS/BEAR from z = log(close/SMA_H)/(σ_15m·√H), deadband ±0.5, σ over a trailing 2880 bars |
| `label` | UP (S_H(t) ≠ BULL ∧ S_H(t+H) = BULL), DOWN (S_H(t) ≠ BEAR ∧ S_H(t+H) = BEAR), NONE |
| `base_rate` | share of UP and of DOWN per fold and half; `events` = run starts per half |

**Validation**:
- the label is NaN where t+H is beyond the data;
- rows are tradeable only (volume > 0, data_ok);
- the truncation test: equal on the overlap at RTOL 1e-9 over 16 cuts.

## Transition screen result (`research/nnmt/screens_transition.json`)

Per (direction, H, level):
- `auc`: early, late;
- `baseline_auc`: base rate (0.5), trailing-trend early and late;
- `margin`: each half;
- `events`: each half;
- `proxy_bp`: top-decile forward-return net of 30 bp, each half;
- `pass`;
- `selected` (the best passing cell per direction).

## Verdict (`research/nnmt/verdicts_<stage>.json`)

Per arm: the `judge_one` output (verdict, arm and control pp, early and late gains, paired, risk,
`folds_changed`), plus `model_share` where relevant and a decision string. The models stage also
records the identity result and the R7 selection.
