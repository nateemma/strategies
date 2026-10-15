# 035 GBR extension: pre-registration

This is committed before any GBR input score. Machine-readable: `preregistration_gbr.json`. The
parents are `preregistration.json` (1f79efa5) and `preregistration_s1b.json` (9b7aa666).

**Model:** GBR (`NNNC_GbbRaw_Open`), the plain LSTM on 8 raw-price features. Its control is
`fs_gbr_gr`, its width book `w_gbr`, and its control sizer `cs_gbr` (new; same model keys as
`fs_gbr_gr`).

**Reproduction, measured before this commit (no input read):**
- GBR's early width-book trades: 494.
- The control-at-share gap: 0.0128, from `cs_gbr`'s f00–f07 picks, all verified with 0 tag
  mismatches.
- GBR's own-decision gap: 0.063 (as in 033).

**S1b-GBR:** the S1b sample, outcome, statistic and rule, unchanged.
- The bars are the same 4,686 candidate run starts as S1b-ARO, with the same outcomes.
- The control is GBR's p_buy, together with bb_width.
- The rule: coverage ≥ 0.9, |ρ_bb_width| ≤ 0.5, |partial| ≥ 0.03, sign ≥ 6 of 8 folds, top 3.

**S2-GBR:** each survivor is added to GBR's 8 raw features as a pre-normalised column; GBR's raw8
scaler is unchanged.
- It passes if its early gap exceeds **both** 0.063 and 0.0128 (so 0.063), with ≤ 2 degenerate
  folds.
- Both references are required, as in 034's S1, because GBR's at-share gap is far below its
  own-decision gap.

**S3-GBR:** the winner, run as `cs_gbr` with only the weights changed, on 15 folds.
- PASS needs judge_one against `cs_gbr` AND `w_gbr`, with `fs_gbr_gr` reported.
- `cs_gbr` is backtested only if S3 is reached.

**Disclosed:**
- S1b-ARO has been read (survivors MS_VR, MS_BV, OF_TRG).
- S2-ARO has been read (no winner: 0.034–0.044 against 0.1003).
- S1b-GBR uses the same bars and outcomes as S1b-ARO.
