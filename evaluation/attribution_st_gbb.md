# Component attribution: `st_gbb` (preset ARM_D_OPEN)

**PROVISIONAL -- not a finding under quality gate 2 (no distant era)**

Full configuration: 9056 trades, +11.04pp per-fold sum.

| component | status | contribution pp | paired verdict | early / late | liveness |
|---|---|---|---|---|---|
| trigger | LIVE | +102.41 | INCONCLUSIVE_ARM_LOSES | INCONCLUSIVE_ARM_LOSES / INCONCLUSIVE_ARM_LOSES | entry_condition_counts.signal moved |
| guard_adx | ALREADY_NEUTRAL |  |  |  |  |
| guard_atr_pct | ALREADY_NEUTRAL |  |  |  |  |
| guard_bb_width | ALREADY_NEUTRAL |  |  |  |  |
| guard_close_norm | ALREADY_NEUTRAL |  |  |  |  |
| guard_guard | ALREADY_NEUTRAL |  |  |  |  |
| guard_rvol | ALREADY_NEUTRAL |  |  |  |  |
| take_profit | LIVE | +12.89 | INCONCLUSIVE_ARM_LOSES | UNCHANGED / INCONCLUSIVE_ARM_LOSES | exit_reason_counts.take_profit moved |
| max_hold | INERT |  |  |  | exit_reason_counts.max_hold |
| roi | LIVE | +1.50 | UNCHANGED | UNCHANGED / INCONCLUSIVE_ARM_LOSES | exit_reason_counts.roi moved |
| exit_signal | ALREADY_NEUTRAL |  |  |  |  |
| stop_atr_band | LIVE | +8.23 | UNCHANGED | UNCHANGED / INCONCLUSIVE_ARM_LOSES | trades_hash moved |
| stop_grace | LIVE | +47.97 | INCONCLUSIVE_ARM_LOSES | WORSENED / INCONCLUSIVE_ARM_LOSES | trades_hash moved |
| stop_volume_confirm | LIVE | -0.23 | UNCHANGED | UNCHANGED / INCONCLUSIVE_ARM_LOSES | trades_hash moved |
