# 035 S1b outcome (recorded at run time, 2026-10-15)

The pre-registration (`preregistration_s1b.json`, commit 9b7aa666) is unchanged. The full results
are in `screen_s1b.json` and `survivors_s1b.json`.

**The sample was smaller than the pre-registration note said:**
- The note gave 17,825 run starts. That count was taken before the pre-registered tradeability
  filter (signal-bar quote volume ≥ 1,000, the width-book simulator's MIN_QV), which was always
  part of the definition.
- Of 17,822 run starts with their entry bar in the test window, 4,686 pass the filter. Per fold:
  261–934.
- The rule was applied exactly as written. Only the stated power was wrong:
  - pooled standard error ≈ 1/√4,686 ≈ 0.015, so the 0.03 cutoff is about 2 standard errors, not
    4;
  - per-fold standard errors are 0.03–0.06, not 0.018–0.027.

**Survivors (top 3 by |partial|, all with |ρ_bb_width| ≤ 0.15):**

| tag | partial | about z (pooled) | sign folds |
|---|---|---|---|
| MS_VR (market volatility regime) | +0.099 | 6.6 | 6/8 |
| MS_BV (BTC 24 h volatility) | +0.093 | 6.2 | 7/8 |
| OF_TRG (global trade-count ratio) | −0.062 | 4.1 | 6/8 |

**Also passing the rule, outside the top 3:** OF_TBG +0.061 (7/8), OF_TRU −0.047 (6/8) and OF_TBU
+0.032 (6/8).

**Observations:**
- MS_VR and MS_BV are both volatility states, so they are probably correlated with each other.
  S2 trains each on its own, as pre-registered.
- OF_TRG's S1 partial on the 412 trades was +0.072, the opposite sign. The per-bar outcome is not
  the traded P&L. S2 and S3 test traded value.
