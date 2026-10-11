# Baseline & evidence log: 026 reversion regime gate

## T001: pre-change baseline (2026-10-11)

```
sha fb66ced
fingerprint NNNC_A_Raw_Open 303456c1fa322b4c
fingerprint SimpleTrigger_Gbb 303456c1fa322b4c
fingerprint SimpleTrigger_FamCombined 27617f430beca1cd
st_fam_combo f00 f9e7a4531e21cce3
data BTC: 2022-01-01 00:00:00+00:00 -> 2026-10-10 15:00:00+00:00 (167324 bars)
data ZEC: 2022-01-13 01:15:00+00:00 -> 2026-10-10 15:00:00+00:00 (166167 bars)
data XRP: 2023-07-14 13:00:00+00:00 -> 2026-10-10 15:00:00+00:00 (113637 bars)
data SOL: 2022-01-01 00:00:00+00:00 -> 2026-10-10 15:00:00+00:00 (167324 bars)
data LINK: 2022-01-14 01:00:00+00:00 -> 2026-10-10 15:00:00+00:00 (166072 bars)
data NEAR: 2022-01-20 01:00:00+00:00 -> 2026-10-10 15:00:00+00:00 (165496 bars)
data AAVE: 2022-01-01 00:00:00+00:00 -> 2026-10-10 15:00:00+00:00 (167324 bars)
data AVAX: 2022-01-01 00:00:00+00:00 -> 2026-10-10 15:00:00+00:00 (167324 bars)
data LTC: 2022-01-01 00:00:00+00:00 -> 2026-10-10 15:00:00+00:00 (167324 bars)
data BCH: 2022-01-01 00:00:00+00:00 -> 2026-10-10 15:00:00+00:00 (167324 bars)
data DOT: 2022-01-01 00:00:00+00:00 -> 2026-10-10 15:00:00+00:00 (167324 bars)
```

XRP's research data starts 2023-07-14 (the 2021–23 delisting gap). Market-wide instruments
need ≥ 6 of 10 pairs, so they stay defined before then (9 pairs).

## T006 — causality on the real data path: PASS

All four instruments give identical values (atol 1e-12) for full history and for history truncated
at 2024-12-21, the end of f07. Every instrument is defined from 2022-01 or 2022-02, long before the
first test fold (2023-01-01).

## T007/T008 — feasibility gate: PROCEED (1 of 4)

See `feasibility.md` / `feasibility.json`. **`volreg` PASSES**, and the `st_gbb` cross-check agrees:

- **Rule (early-chosen):** off if volreg < 1.033 (p33), i.e. reversion is off in calm markets.
- **Off-state reversion P&L** is −14.65 early and −11.79 late.
- **On-state** is +38.10 early and **−12.91 late**.
- On-share early 67%. Off bars 55% / 59%. Late longest off run 3%, so not a calendar switch.

`breadth`, `persist` and `btctrend` STOP. In each, the off state was profitable early, so the
instrument carries no early information about reversion's losers.

**Caveat for US2.** The "on" state still loses late (−12.91). On the bucket split, the gate removes
about half the reversion late loss, not all of it. Beating momentum alone (+2.50 late) is therefore
unlikely, but per FR-004 the walk-forward arm decides.

## T013/T014 — controls and gated arm

| arm | vs `st_fam_combo` | total | early | late | liveness |
|---|---|---|---|---|---|
| `st_reg_off` (gate None, 64-bar startup) | **15/15 identical** (FR-006) | +15.39 | +35.76 | −20.37 | — |
| `st_reg_on4096` (gate None, 4096-bar startup) | 11/15 identical, 4 differ slightly | +15.54 | +35.92 | −20.38 | — |
| `st_reg_volreg` (gate, 4096 startup) | — | **+38.14** | +44.19 | −6.05 | 80,422 rows refused, 4,463 retagged, 820,908 off bars |

**H1 control outcome.** The longer startup alone changed 4 folds, so `st_reg_on4096` is the
always-on comparison (pre-declared rule). Before the chain, a single-fold smoke test (f00)
confirmed that the whitelist and `get_pair_dataframe` plumbing works inside a backtest: 3,649
refusals in f00 alone.

## T016 — stage decision: NOT adopted

The late verdict against the bar `st_bgo_mx` is INCONCLUSIVE_ARM_LOSES: −8.56pp, CI [−20.6, +10.6],
p 0.53. Against the previous arm `st_reg_on4096` it is +14.32pp, p 0.13, reported only.

## T020 — 023–025 safety re-check: PASS

These are fresh `run_arm` f00 re-runs under the 026 code, with result files timestamped
2026-10-11 21:43:
- `st_fam_combo` gives `f9e7a4531e21cce3` ✓.
- `st_gbb` gives `aa058ba815f48627` ✓.

Fingerprints are unchanged: arm D and `SimpleTrigger_Gbb` are `303456c1fa322b4c`;
`SimpleTrigger_FamCombined` is `27617f430beca1cd`.
