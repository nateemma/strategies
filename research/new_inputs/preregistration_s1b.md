# 035 S1b pre-registration: the screen on every early gbb candidate run start

This is follow-up 1 of `report.md`, committed before any S1b sample outcome or score was
computed. Machine-readable: `preregistration_s1b.json`. The parent pre-registration is
`preregistration.json`, commit 1f79efa5.

**Why:** S1 scored the 17 inputs on the 412 early width-book trades, about 50 per fold. At that
size a fold's partial correlation has a standard error near 0.14, so the sign-in-6-of-8-folds rule
had little power. S1b asks the same question on every early gbb candidate run start: 17,825 bars,
1,344–3,142 per fold.

**Sample:** in folds f00–f07, the first bar of each run of the width book's gates *without* the
width cut (the gbb rule plus the control's guards). Each pair is taken separately; the entry bar
must lie in the fold's test window and quote volume must be ≥ 1,000.

**Outcome:** the net profit ratio of a trade entered at the next open and exited by the control's
24 h-grace exits, the same replay as the width-book simulator. It ignores occupancy, the wallet and
`max_open_trades`.

**Statistic:** the partial Spearman correlation with the outcome, given bb_width and the control's
p_buy. It is computed pooled and per fold.

**Rule:** an input survives only if all of these hold:
- coverage ≥ 0.90;
- |ρ with bb_width| ≤ 0.5;
- |pooled partial| ≥ 0.03, which is about 4 pooled standard errors;
- the same sign in ≥ 6 of 8 folds (per-fold standard error 0.018–0.027).

At most 3 survive, ranked by |partial|. Survivors go to the parent's S2 and S3, unchanged.

**Disclosed:**
- **S1 was already read** (no survivors). Its near misses were OF_TRU +0.159 (5/8), OF_GUS −0.142
  (4/8) and MS_BV +0.096 (6/8).
- **Overlap:** the width-book trades' signal bars are a subset of this sample's candidate bars.
- **Same rule for every input:** it applies to all 17 identically.
- **The outcome is per bar,** one step further from traded P&L than S1, so S2 and S3 remain the
  tests of traded value.
