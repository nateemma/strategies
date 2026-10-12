# Liveness: 029 declared switches on an NNNC arm (T028–T030)

Each arm is arm D (`NNNC_A_Raw_Open`) on its own borrowed per-fold weights and per-fold thresholds, with ONE declaration added. Undeclared, arm D reproduces its baseline 15/15 (verification.md): every switch is inert when not declared.

| switch (declaration) | control | decisions changed | trades | `trades_hash` | framework_sha |
|---|---|---|---|---|---|
| **C2 ATR take-profit** for reversion (`FAMILY_EXIT_OVERRIDES={"reversion": {"take_profit_atr_mult": 2.897…}}`) | arm D f00 | exits: {'roi': 2, 'take_profit': 19, 'trailing_stop_loss': 26, 'unclog_12h': 4} vs {'roi': 1, 'take_profit': 23, 'trailing_stop_loss': 25, 'unclog_12h': 3} | 51 vs 52 | differs | `409ebd662dfd2e90` vs `303456c1fa322b4c` |
| **Regime gate** (`REVERSION_REGIME_GATE` = st_book_rev2's volreg cut, startup 4096) | `NNNC_Live_GateOff` f00 (gate off, same startup) | 175 reversion-tagged signal rows refused in the off state (70614 off bars) | 31 vs 52 | differs | `b03fa4a267305a28` vs `ab100aa13e42715d` |
| **Trigger AND model** (`ENTRY_MODE="trigger_and_model"`, `TRIGGER_NAME="gbb"`) | arm D, all 15 folds | 2 of 13608 model-signal rows refused (off-trigger) | identical 15/15 | identical 15/15 | `545339b237ec7190` vs `303456c1fa322b4c` |

- **C2 and the gate are live** on the decisions they own: exits for C2; entries refused for the gate.
- **The gate-off control matters.** The 4096-bar startup alone changes arm D's f00 trades (`NNNC_Live_GateOff` vs arm D: same count, different `trades_hash`), so the gate is judged against its own startup.
- **The trigger gate is live but nearly a no-op on arm D.** It refuses exactly the model signals the gbb trigger doesn't cover. There are only 2 of them in 15 folds (f06, f11), and neither became a trade, so trades are identical.

## Correction: arm D's model enters almost only on gbb bars

The 023 report's caveat, "only 43.5% of arm D's trades are also twin trades … the model enters *outside* the `gbb` trigger more than half the time", was a misreading.

- **What 43.5% actually measured:** a trade-set overlap with the twin (`SimpleTrigger_Gbb`). The twin takes about 9,000 trades against arm D's 1,400, so one-trade-per-pair occupancy alone makes the two trade sets diverge.
- **The direct measurement:** 2 of 13608 arm D model-signal rows (0.015%) fall outside the gbb trigger.
- **This fits the label.** Arm D trains on `gbb_path`, whose positives require the gbb band condition.
- **Conclusion:** arm D's model is *selection within the gbb trigger*. The planned follow-up "what are arm D's out-of-trigger entries" has nothing to study, and trigger-AND-model is not a distinct arm for arm D.

### Per fold (trigger AND model vs arm D)

| fold | arm D trades | trigger∧model trades | trades | model-signal rows | refused (off-trigger) |
|---|---|---|---|---|---|
| f00 | 52 | 52 | = | 304 | 0 |
| f01 | 22 | 22 | = | 340 | 0 |
| f02 | 21 | 21 | = | 247 | 0 |
| f03 | 104 | 104 | = | 876 | 0 |
| f04 | 130 | 130 | = | 792 | 0 |
| f05 | 93 | 93 | = | 993 | 0 |
| f06 | 54 | 54 | = | 814 | 1 |
| f07 | 249 | 249 | = | 2057 | 0 |
| f08 | 218 | 218 | = | 2148 | 0 |
| f09 | 46 | 46 | = | 782 | 0 |
| f10 | 22 | 22 | = | 324 | 0 |
| f11 | 226 | 226 | = | 1718 | 1 |
| f12 | 59 | 59 | = | 807 | 0 |
| f13 | 87 | 87 | = | 1066 | 0 |
| f14 | 37 | 37 | = | 340 | 0 |
