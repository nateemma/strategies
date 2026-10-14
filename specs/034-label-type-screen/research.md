# Research: Label-Type Screen for NNNC (034)

All Technical Context unknowns are resolved below. Each item: Decision, Rationale, Alternatives.

## R1. Sell side per candidate (found while planning)

Every candidate trains a three-class model, so each needs a SELL label. `get_train_sell_signals`
has a dedicated sell variant for forward_mae, triple_barrier, future_expectancy, quantile_future,
local_min_followthrough, breakout_gbb, gbb_path and gbb_fwdmae.

**mfe_mae_ratio has none.** It falls through to the BUY labeler, so its raw SELL labels equal its
BUY labels. Training then applies conflict resolution ("sells override buys") after the sell-side
peak filter, which keeps only sell peaks followed by a real decline. The surviving BUY and SELL
classes are therefore whatever the filter leaves, not a mirror of the buy rule.

- **Decision**: no Framework change. A pre-training **label audit** (stage S0a, no model, no P&L)
  computes every candidate's BUY / SELL / HOLD shares after the peak filter and conflict
  resolution, on the training window of every early fold. A candidate whose kept BUY share is under
  1% of training rows in any fold is INCONCLUSIVE and is not trained. The audit also reports the
  raw BUY–SELL overlap, so mfe_mae_ratio's behaviour is on record.
- **Rationale**: the spec varies the label only, using each labeler's own sell handling (FR-004).
  Writing a new sell variant would add a new label definition, which is out of scope. The audit is
  cheap (labels only) and runs before the pre-registration is committed, so it can stop a
  degenerate candidate without reading any outcome.
- **Alternatives**: add `labels_mfe_mae_ratio_sell` (a new label definition, out of scope; can be
  a follow-up); train mfe_mae_ratio with no SELL class (breaks the one-variable rule).

## R2. Declared label parameters

Every candidate receives the control's base parameters (horizon 48, min_gain 0.02, min_loss 0.02,
rvol_threshold −1 on the buy side). Each labeler keeps only the keys it accepts. Declared extras go
through `LabelParamsMixin.LABEL_PARAMS`, where an unsupported key raises. The rule, fixed before any
training: **labeler defaults, plus the control's horizon and ±2% thresholds, plus stop-matched
adverse-excursion caps (0.04, as GFM) where a labeler has one.**

| label | id | declared extras (buy; sell mirrors unless noted) | prior reference |
|---|---|---|---|
| gbb_path (control) | 27 | none (existing weights) | `wf_aro_g24` |
| gbb_fwdmae | 28 | as GFM: `LABEL_MAX_DRAWDOWN` 0.04 (existing weights) | `NNNC_GbbFwdMae_Open` |
| forward_mae | 0 | `max_drawdown` 0.04 | GFM's stop-matched cap |
| triple_barrier | 1 | buy `pt` 0.02, `sl` 0.05 (gbb_path's own target and stop); **sell declared explicitly** `pt` 0.05, `sl` 0.02 | `NNNC_Oracle_TripleBarrier_H12` used pt = sl at H12 |
| mfe_mae_ratio | 3 | none (`min_ratio` 2.0 default) | `NNNC_Oracle_MfeMaeRatio` |
| future_expectancy | 12 | none (`min_expectancy` 0.01 default) | `NNNC_Oracle_FutureExpectancy` |
| quantile_future | 2 | none (`top_quantile` 0.8 default) | — |
| local_min_followthrough | 4 | `max_drawdown` 0.04 | `NNNC_Oracle_LocalMinFollowthrough` |
| breakout_gbb | 24 | none (guard 0.2, width 0.035 defaults) | `NNNC_BreakoutOpen` |

- **triple_barrier's sell side is not mirrored by reusing the buy values.** In
  `labels_triple_barrier_sell`, `sl` is the DOWN barrier and `pt` the up barrier, so reusing
  `{pt 0.02, sl 0.05}` would label "−5% before +2%". The mirror of the buy rule (+2% before −5%)
  is "−2% before +5%": `LABEL_PARAMS_SELL = {"pt": 0.05, "sl": 0.02}`. The training call's
  `min_loss` 0.02 raises `sl` to at least 0.02 and the buy side's `min_gain` 0.02 raises `pt` to at
  least 0.02, so neither changes these values.
- **Sell-path key handling**: the explicit sell branches in `get_train_sell_signals` filter params
  to a hand-written `allowed` set and drop anything else silently (strict keys are honoured only on
  the fallback). So each declared sell key is checked against its branch's `allowed` set (T003), and
  S0a's liveness diff catches a key that is accepted but has no effect.
- **Liveness**: S0a records, per candidate, the labels with and without each declared extra on
  f00's training window. A declared extra that differs from the labeler's default but changes no label is an S0a failure
  (Constitution IV); an extra equal to the default (e.g. TB sell `sl` 0.02) is recorded as such.
- **Alternatives**: labeler defaults only (horizon 36, 0.7% gains) would change two variables at
  once against the control; tuning per label is out of scope.

## R3. Classes and model keys

- **Decision**: one generated class per candidate, `NNNC_LS_<TAG>` =
  `LabelParamsMixin + NNNC_A_Raw_Open` with `TRAINING_TYPE` and `LABEL_PARAMS` set, a 4-decimal
  `prediction_threshold`, and the control's `.json` copied. Its sizer twin
  `NNNC_LS_<TAG>_SZ` = `WidthSizerMixin + WidthBookMixin + NNNC_LS_<TAG>` with the width book's
  `WIDTH_CUTS`.
- `TRAINING_TYPE` and `LABEL_PARAMS` are model-key attributes; `WidthBookMixin` and
  `WidthSizerMixin` attributes are ignored, so **the entry-source class and its sizer twin resolve
  to the same model key per fold** and are trained once. A test asserts this, plus that the
  control's key is unchanged.
- **Amendment (implementation, before any training): the entry family is pinned.** Feature 029
  derives a model's entry family from its label: outcome-only labels (forward_mae, triple_barrier,
  mfe_mae_ratio, future_expectancy, quantile_future, local_min_followthrough) have NO family, so
  their entries would be untagged, and breakout_gbb is momentum, which turns the held-pair add
  off. Each would be a second variable against the control (resolved family reversion, add on).
  The generated classes declare `MODEL_ENTRY_FAMILY = "reversion"`; a test asserts family,
  held-pair add and family exits equal the control's (entry-source classes) or the width book's
  (sizer twins, where the width book itself turns the add off).
- Tags: `FMAE`, `TB`, `MMR`, `FEXP`, `QF`, `LMF`, `BOG`. gbb_fwdmae reuses GFM (`NNNC_GFM_fNN`);
  its sizer twin is `NNNC_LS_GFM_SZ` on `NNNC_GbbFwdMae_Open`.
- **Alternatives**: one class with a runtime label switch (hides the label from the class name and
  record).

## R4. Reusing existing weights

- **gbb_path**: the control (`wf_aro_g24`, classes `NNNC_ARO_fNN`). Its S1 gap 0.072 is the
  pre-registered reference; its tables are `research/nnnc_framework/signals/ARO_fNN.parquet`.
- **gbb_fwdmae**: GFM on the same LSTM_RAW architecture (`sw_gfm_ctl_b030`), with signal tables
  already on disk (`GFM_fNN`). S1 needs no training or inference. If it survives, its arms are
  built on `NNNC_GbbFwdMae_Open` with the ARO control's 24 h-grace params snapshot
  (`params_20261013-g24_NNNC_A_Raw_Open.json`; GFM inherits `NNNC_A_Raw_Open`, so every key in it
  applies) and the `sw_gfm_ctl_b030` folds and dates. Weights are found by model key, which must
  equal GFM's existing fold keys (tested at f00 and f07).
- **Caveat recorded**: GFM's weights were trained before the 24 h grace change; grace is an exit
  parameter and not in the model key, so the weights are the same ones a retrain would produce
  (030 key semantics).

## R5. Training chain and cost

- **Decision**: reuse the 032 driver (`research/candidate_training/train.py`) with runtime plan
  registration, as 033 did, looping candidates in pre-registered order, folds f00–f07 only.
  Resumable; refuses to start while another chain runs.
- **Cost**: ARO trains about 12 min per fold, so 7 × 8 ≈ 11 h, in one sequential chain (probably
  overnight). Signal tables: 56 test-window inference passes. Picks: 56 single-fold sizer picks plus
  up to 24 entry-rate picks for survivors.
- **f00 smoke first**: one candidate (forward_mae), f00 only, to check training, key, table and
  pick end to end. Its record is set aside by name if any arm record is written.

## R6. S1 instrument

- **Decision**: reuse 033's `separation.py` logic, generalised to a list of candidates: the width
  book is simulated once per fold from the control's tables (cuts from `width_twin.json`), so every
  candidate re-weights the identical trades. A candidate's agree = BUY argmax and p_buy above its
  fold threshold, picked at a 0.5 agree share on the first bar of each width-book candidate run in
  the validation tail (033 `pick.py`, one fold per process, `ft_bot_start()` applied).
- **Fairness**: the control's pre-registered 0.072 uses its own entry decision, not a 0.5 share.
  So the control is also scored at a 0.5 share with the same picker, and **a candidate must beat
  both** the pre-registered 0.072 and the control's 0.5-share gap.
- **AUC**: within-candidate AUC of p_buy against trade pp > 0, on the same trades; reported, not
  gating.

## R7. Entry-source threshold

- **Decision**: as 032 Q1 and as the control's own thresholds were picked,
  `evaluation/pick_threshold.py --mode pick --target-rate 0.014735` per fold on the training window
  (not the validation tail used for the sizer's agree share), with ineligible bars counted as zero. Stored at 4 decimals in the
  plan; loading is verified (`--verify`-style check) before the backtest.
- **Rationale**: the same overall entry rate as the control keeps trade counts comparable.
- **Alternatives**: run-matched thresholds (032 follow-up 1; recovered counts but not quality).

## R8. Survivor rule and S2 judgement

- **Survivors**: candidates passing S0 whose early gap beats both references (R6); the top 3 by
  gap. Ties broken by AUC. The output (`survivors.json`) is committed before any survivor backtest.
- **S2**: each survivor in two roles (breakout_gbb sizer only), early folds f00–f07, against
  `w_aro` and `wf_aro_g24`. Pass = early pp above both; a minimum margin under 1 pp triggers the
  paired seed-2 repeat (retrain at seed 2 on f00–f07, both seeds positive).
- **S3 winner**: the passing arm with the largest minimum margin.

## R9. S3 and the GBR repeat

- The winner is trained on f08–f14 (first read of the late half for this label), thresholds picked
  the same way, and judged with `judge_one` vs the control plus late pp vs the width book.
- On a pass, the GBR repeat builds the same label on `NNNC_GbbRaw_Open` (control `fs_gbr_gr`,
  width book `w_gbr`) for all 15 folds, same rules.

## R10. Leakage check

- Candidates change only the label. The check is that each candidate's feature column list equals
  the control's (no `%train_*` or forward column added), recorded per candidate at S0a, and that
  thresholds come from the validation tail or training window only.

## R11. Byte-identity

- Existing classes are untouched; new code is new files plus runtime registrations in research
  scripts. SC-003: two control folds (f00, f07) re-backtested give the recorded trades hash.
