# NNNC switch experiment — report (2026-10-12)

*PROVISIONAL.* The pre-registration is [preregistration.md](preregistration.md) (`0efad08`, committed
before any arm ran); the full tables are in [verdicts.md](verdicts.md).

**Result: neither switch is adopted for NNNC.** All NNNC defaults stay as they are.

| switch | arm D (ARO) | GBR | GFM | MLX | decision |
|---|---|---|---|---|---|
| **C2** ATR take-profit | PASS (+7.27 / +0.62) | FAIL (−3.53 / +3.95) | FAIL (−0.65 / −4.06) | PASS (+3.38 / +0.31; paired IMPROVED, p 0.042) | NOT ADOPTED (arm D + 1/3) |
| **Gate** volreg | FAIL (−0.81 / −6.21) | FAIL (+5.44 / −7.85) | FAIL (−3.21 / −0.08) | FAIL (−3.42 / −1.59) | NOT ADOPTED (arm D + 0/3) |

Cells give the early / late gain in pp (Σ per-fold `return_pct`, arm − control). Every paired and
risk verdict is UNCHANGED, except MLX C2's paired verdict, which is IMPROVED.

## What this says

- **C2's effect depends on the model.**
  - It helps the two `gbb_path` models with the default exit: arm D +7.89 pp in total, though the
    late half is only +0.62; MLX +3.69 pp.
  - It is mixed on GBR, which shares arm D's label but takes about 1.7× the trades.
  - It hurts GFM, whose `gbb_fwdmae` label already selects entries for low forward adverse
    excursion.
  - On every model C2 swaps fixed take-profit exits for ROI exits (arm D 824 → 586 take-profit,
    37 → 177 ROI). Its k was fitted on SimpleTrigger's entries, not on a model's.
- **The gate hurts every model.** It removes 37–51% of trades. Net of its own 4096-bar startup
  control, every model loses in total, and 3 of 4 lose in the late half: the half where the gate
  helped gated reversion on SimpleTrigger (026). One hypothesis, not tested here: the model's
  selection within the gbb trigger already avoids much of what the volreg off-state removes, so
  the gate mostly drops good entries. Measuring that overlap would be the check.
- **The startup alone is neutral:** within ±0.5 pp per half on every model.

## Instrument checks

- **No training:** 0 × `training on` in the chain log.
- **Liveness reproduced:** `sw_aro_c2` f00 = 029's liveness run `nnnc_live_c2` f00 (same
  `trades_hash`).
- **Same-day native controls:**
  - GBR and GFM = their 10-08 records, 15/15.
  - MLX has the same 378 trades but only 5/15 hashes match its 09-09 record, which predates the
    stop fix. As planned, the fresh control is the comparator.

## Not run

- **The combination:** the rule requires both switches to be adopted.
- **Trigger-AND-model:** a no-op on these models (029).

## Fix-oriented follow-ups (not run)

1. **C2 per model:** re-fit k on each model's own trades, inside its training window only, and
   judge walk-forward. A take-profit set from SimpleTrigger's entry distribution may not suit a
   model's selected entries.
2. **Gate diagnosis before any retuning:** measure how far the model's accepted rows and the
   volreg off-state overlap, and the P&L of the gated-out model trades, by half.
3. **MLX C2** passed on its own (paired IMPROVED early). It can be declared on `NNNC_MLX`
   individually if wanted, but the pre-registered rule does not make it a family default.
