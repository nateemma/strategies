Width-relative label for NNNC (feature 033).

Background:
- The model check (research/model_calibration/findings.md §1–7) found that on gbb-rule-eligible bars
  (guard_metric < 0.3 AND bb_width > 0.035) the NNNC models' ranking is largely a bb_width sort.
- A model-free width book (NNNC/_width_book.py; arms w_aro/w_gbr/w_gfm/w_mlx) matches or beats every
  model book in real backtests.
- Feature 032 (research/candidate_training/report.md) trained on eligible rows only:
  - within-candidate AUC rose (GBR +0.031 in both halves);
  - the CO scores became LESS width-correlated (Spearman 0.58 → 0.30 for GBR);
  - but the books failed S2, with ~40% fewer trades;
  - and as a sizer on the width book, CO was worse than the current model.
- The label (gbb_path, TRAINING_TYPE 27) still rewards what width already explains.

Goal: a label that asks "did this candidate do better than its width would predict?", so the model
learns information orthogonal to width. That model is then used where width cannot help: ranking and
filtering inside the width book.

Mechanism:
- A new label method (next LabelMethod id after gbb_fwdmae = 28) in Framework/TrainingSignals.py. It
  stays strategy-agnostic: the outcome variable and the width reference are parameters, not
  hard-coded NNNC logic.
- Outcome: a forward, path-aware outcome per eligible bar over the existing horizon (default H48).
  The proposed default is the realised return of the gbb_path trade: the target or stop as first
  hit, else the close at the horizon. Confirm in the spec.
- Width reference: the expected outcome for the bar's bb_width quantile bucket, estimated on the
  model's OWN training window and pooled across pairs.
  - The proposed default is 10 equal-count buckets, refit per fold. Confirm in the spec.
  - Pooling across pairs needs a training-window hook, not a per-pair label function: decide where
    it lives without putting strategy code under Framework/.
- Label: BUY if the outcome exceeds its bucket's expectation (or a declared margin above it), else
  HOLD; no SELL. This is roughly 50% positive within every width bucket by construction, so width
  alone carries ~no label information.
- Training uses the 032 row filter (CandidateTrainingMixin, TRAIN_ROW_FILTER on gbb-eligible rows).
- The new label attributes join MODEL_KEY_ATTRIBUTES only on the new arms' mixin, so every existing
  model key, fingerprint and per-fold trades_hash stays unchanged.

How the model is used (arms), proposed:
- (A) Width book filter: width book AND the model fires.
- (B) Width book sizer: agree 1.0 / disagree 0.5, run as the simulator screen first.
- Option: (C) the model as entry source, 032-style, for comparability only. Decide in the spec
  whether (C) is in scope.
- Comparators: the width book (w_aro / w_gbr) and the 24 h controls (wf_aro_g24 / fs_gbr_gr).
- Models: ARO (NNNC_A_Raw_Open, LSTM_RAW) and GBR (NNNC_GbbRaw_Open, raw8) first, 15 walk-forward
  folds each.

Thresholds (lessons from 032 follow-ups):
- prediction_threshold is DecimalParameter(decimals=2), so picked thresholds are rounded on load.
  Either declare thresholds on the 0.01 grid, or decide (user's call) whether to raise decimals for
  the new arms only.
- Any script that scores fold strategies outside backtesting must call ft_bot_start(), or the fold
  .json params are not loaded. The parameter objects are class-level, so a value set in one load
  leaks into the next.
- Match the operating point on entry runs, not bar rate, on the validation tail
  (research/candidate_training/run_match.py). For a filter arm (A), the operating point is instead
  the share of width-book trades kept; pre-register it.

Evaluation, staged and pre-registered BEFORE any model is trained:
- S0 (instrument):
  - label sanity per fold: positive rate per width bucket ≈ uniform;
  - Spearman of the label with bb_width ≈ 0 on eligible bars;
  - simulator fidelity ≥ 0.95;
  - 0 entry violations.
- S1: within-width-book separation of the retrained model must beat the current model's in both
  halves. The measure is the agree − disagree pp/trade gap from research/candidate_training/
  co_sizing.py; the current gaps are ARO 0.072 / 0.085, GBR 0.063 / 0.072.
- S2: the early half (f00–f07) of arm A vs the width book and the control.
- S3: the late half (f08–f14), confirmatory, judged with judge_one against both.
- The small-margin rule and the paired seed-2 repeat as in 032.
- A pass is PASS (PROVISIONAL), never adopted. Confirmation route: folds after 2026-09-07, or
  dry-run.

Out of scope:
- Changing the gbb rule or the width book itself.
- Regime-conditional labels.
- GFM/MLX (optional later).

Constraints:
- Default OFF and byte-identical when off.
- Framework/ stays strategy-agnostic.
- Parameter dicts merge from the parent.
- No pair-specific code.
- One trainer/backtest chain at a time, with the threshold pick one fold per process (a 15-fold
  pick process was reaped for low memory).
- The exit-parameter guard test keeps passing.
- The live dry-run bot is untouched.
- Promoting anything to a default is the user's decision.
