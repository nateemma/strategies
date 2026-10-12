# Pre-registration: NNNC framework switches (2026-10-12)

**Committed before any switch-arm backtest.** The machine-readable copy is
[preregistration.json](preregistration.json); `judge.py` reads only that file, and refuses to run
if the file has uncommitted changes or any arm record predates its commit.

**Question:** do the two framework switches adopted on SimpleTrigger carry over to NNNC's model
entries?

- **C2:** 028's reversion ATR take-profit, 2.897 × signal-bar ATR, clamped to [1%, 5%].
- **Gate:** 026's volreg reversion regime gate, cut 1.3658, judged against a control with the same
  4096-bar startup.

Trigger-AND-model is not tested: it is a proven no-op on arm D (2 of 13,608 signal rows, 029).

## Arms

Every arm runs on the model's own per-fold weights (borrowed, never trained) and its per-fold
thresholds over f00–f14 (2023-01 → 2026-09). Every model's label is reversion-family.

| model | base class | weights | control |
|---|---|---|---|
| **ARO (primary, arm D)** | `NNNC_A_Raw_Open` | `NNNC_ARO` | `wf_aro_a029` (after the merge, 2026-10-12) |
| GBR | `NNNC_GbbRaw_Open` | `NNNC_GBR` | `sw_gbr_ctl` (same-day native re-run) |
| GFM | `NNNC_GbbFwdMae_Open` | `NNNC_GFM` | `sw_gfm_ctl` |
| MLX | `NNNC_MLX` (guards on) | `NNNC_WF` | `sw_mlx_ctl` |

Each model also has three switch arms:
- `sw_<m>_c2` (`NNNC_SW_<M>_C2`), judged against the model's control;
- `sw_<m>_gate` (`NNNC_SW_<M>_Gate`), judged against `sw_<m>_gateoff` (`NNNC_SW_<M>_GateOff`).

## Rule (fixed here)

**Gain** = Σ per-fold `return_pct` (arm) − Σ (control), computed separately over the early half
(f00–f07) and the late half (f08–f14).

**Per model, PASS** requires all of:
- early gain > 0, **and**
- late gain > 0, **and**
- the 15-fold paired verdict maps to pass, **and**
- both 15-fold risk verdicts (Sharpe, maxDD) map to pass.

The verdict maps are 028's:
- WORSENED fails;
- IMPROVED, UNCHANGED, INCONCLUSIVE_UNDERPOWERED and INCONCLUSIVE_ARM_LOSES pass;
- ABSORBED and UNRESOLVABLE are inconclusive.

**INCONCLUSIVE** if any of these holds:
- the control has fewer than 200 trades;
- the arm changed trades in no fold;
- a mapped verdict is inconclusive.

**Switch ADOPTED for NNNC** only if it PASSes on ARO **and** on at least 2 of GBR, GFM, MLX. The
combination is run only if both switches are ADOPTED.

**Label:** PROVISIONAL. The values were tuned on SimpleTrigger trades over these same folds, so
this tests whether the effect carries over to model entries, not its out-of-sample value. 2022
cannot be used: it is inside every model's training window.

**Reported only:**
- per-half paired and risk verdicts;
- the startup effect (GateOff vs control);
- exit-reason counts;
- whether each same-day native control still matches its earlier record.
