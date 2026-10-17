# Forward confirmation of cs_cmp / cs_mix: report

**Sources:** `preregistration.json` (commit 655f3304, before any forward run); `build.json` (composite
columns: 0 mismatches vs the 037 cache on all 10 pairs); `compare.json`. Run 2026-10-17. Window
2026-09-07..10-12, fold f14 frozen (fold id f15 = cs_aro's fwd1). A blind sign check, not a test.

## Read: both INCONSISTENT

| arm | trades (identical to width) | wallet % | minus width (+0.296) | paired USDT vs width, 90% CI | minus cs_aro (−0.116) | paired USDT vs cs_aro, 90% CI | agree share |
|---|---|---|---|---|---|---|---|
| cs_cmp | 49 (yes) | +0.075 | −0.22 | −22 [−75, +31] | +0.19 | +19 [−36, +79] | 0.59 |
| cs_mix | 49 (yes) | −0.327 | −0.62 | **−62 [−113, −14]** | −0.21 | −21 [−57, +7] | 0.59 |

- On these 49 trades, the composite's agreeing trades did better than its disagreeing ones (width-book
  return per trade 0.67% vs 0.29%); the mixed score's did worse (0.13% vs 1.07%), as did ARO's
  (0.41% vs 0.64%). The mix inherits ARO's wrong-way ranking on this window.
- cs_mix's interval vs the width book excludes 0: on the only blind data, the mixed sizer lost.
- Note: by enter_tag, cs_aro agreed on 27/49 here; the 2026-10-16 note recorded an agree share of 0.20
  for this window -- not re-derived, flagged.
