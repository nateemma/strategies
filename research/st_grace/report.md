# SimpleTrigger grace study — report (2026-10-13)

Question: does NNNC's adopted 24 h grace window (`ed6049b`) also suit the model-free SimpleTrigger
framework? Pre-registration `dec1d0f` ([preregistration.json](preregistration.json)). Judge:
`judge.py` → [verdicts.json](verdicts.json). **PROVISIONAL.** 2022 (3 folds) is the confirmatory
window. 2023–26 is exploratory, because its late-half replays were seen in [screen.json](screen.json).

## Result: not adopted on either family; SimpleTrigger presets stay at 6.5 h

| family | reference → arm | 2023–26 early / late | 2022 (3 folds) | decision |
|---|---|---|---|---|
| MOM | `st_bgo_mx` → `st_g24_mom` | FAIL +12.87 / **−3.20** (total 25.19 → 34.85) | PASS +15.45 (−58.01 → −42.55) | **NOT ADOPTED** |
| REV | `st_prom_c2` → `st_g24_rev` | FAIL +8.87 / **−0.43** (total 63.06 → 71.50) | **FAIL −6.10** (−0.47 → −6.57) | **NOT ADOPTED** |

- All paired and risk verdicts are UNCHANGED or INCONCLUSIVE_ARM_LOSES. Momentum maxDD over
  2023–26 rises from 18.2% to 24.1%.
- Reversion 2022 fails, as 028's 12 h grace (C4) did on 2022: stopped reversion trades recover less
  in 2022.
- Momentum gains in 2022 only by losing less, in a window where the momentum book loses 58 pp
  either way.

## Why the replay was so far off: one trade per pair

The screen predicted reversion +47.3 / +21.3 pp (fidelity 99.8%); the arm delivered +8.9 / −0.4. The
arm's gain splits exactly into three parts:

| | same entries in both runs (Δ pp) | control entries the arm did not take (n, pp forgone) | new arm entries (n, pp) | net = judged gain |
|---|---|---|---|---|
| REV early | +28.50 | 572, +26.95 | 86, +7.32 | +8.87 |
| REV late | +10.95 | 385, +10.18 | 42, −1.21 | −0.44 |
| MOM early | +14.14 | 727, +5.22 | 235, +3.94 | +12.86 |
| MOM late | −9.70 | 504, −5.49 | 140, +1.00 | −3.21 |

- **Of the entries the arm did not take, 99% were blocked because the arm still held a trade on
  that pair** (REV 944 / 957, worth +37.2 pp; MOM 1,203 / 1,231). The 10 open slots were never all
  full.
- Longer grace improves the reversion trades it keeps, but it holds the pair through the window in
  which the next profitable reversion signal arrives.
- This is 024's finding ("pair-level blocking, not exits, is the lever") seen from the exit side.

**NNNC pays the same cost.** The same split of round 1's adopted 24 h arms gives:
- arm D: +19.25 on the same entries, 263 entries forgone (261 pair-held) worth +13.60;
- GBR: +11.41 on the same entries, 432 forgone (427 pair-held) worth +7.02.

24 h grace still won on NNNC, but roughly 40% of its gross gain went to blocked re-entries.

## Implications

1. SimpleTrigger and NNNC now differ in default grace (6.5 h vs 24 h). For framework experiments
   meant to transfer to NNNC, run SimpleTrigger at 24 h explicitly (the `_G24` classes); its own
   default stays at 6.5 h.
2. The framework lever is **what happens when a fresh entry signal fires on a pair that is still
   held**. On reversion that is worth up to +37 pp in 2023–26, and on NNNC arm D up to +13.6 pp.
   It is a framework change (exit / position policy), not a grace value.

## Tools

`screen.py` (grace-grid replay on the reference books), `judge.py` (2023–26 rule plus 2022
persistence), and the `_G24` arm classes.
