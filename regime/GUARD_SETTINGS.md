# Recommended guard settings, per family — and the evidence for each

Verified 2026-09-07 by loading each strategy in an ISOLATED process (see the
warning at the bottom — loading two in one process gives wrong answers).

## The recommended values are the CURRENT values. Both families are already at their measured optimum.

| parameter | **NNNC_MLX** | **NNMT_MLX** | evidence |
|---|---|---|---|
| `entry_rvol_threshold` | **2.0** | **0.2** | The families' sharpest difference — 10x. NNNC: rvol is the DOMINANT guard, profitability cliff between 2.0 and 1.0; 2.5 was tested and rejected (PF fell in all 3 windows); 3.1 was overturned by the trailing-stop repair. NNMT: nearly INERT at 0.2 (removing it moves 176 → 174 trades) |
| `entry_adx_threshold` | **38.0** | **38.0** | NNMT: the DOMINANT filter — removing it takes 176 → 542 trades on W1 (3.1x) |
| `entry_atr_pct` | **0.006** | **0.006** | NNMT: second strongest — 176 → 392 (2.2x) |
| `entry_bb_width_threshold` | **0.06** | **0.06** | NNMT: mild — 176 → 234 |
| `entry_guard_threshold` | **0.3** | **0.3** *(INERT)* | NNMT: **byte-identical trades in all three windows** when neutralised — it does nothing. NNNC: saturating plateau, 0.6 byte-identical to 0.3. **Do not spend hyperopt budget on this parameter in either family** |
| `entry_close_norm_threshold` | **−0.1** | **−0.1** | NNMT: mild — 176 → 205 |
| `entry_enable_guards` | **True** | **True** | Also gates the exit-side guard block and the `volume > 0` sanity check — never disable it to express "guards off" |
| `prediction_threshold` | **0.69** | **0.5** | NNNC's per-window optima disagree (0.82/0.80/0.75) and do NOT transfer. NNMT: 0.4–0.6 are largely byte-identical on W1, so the value is not critical there |
| `enable_exit_signal` | **False** | **True** | **Family-specific, opposite verdicts.** NNMT +0.78pp on W1, positive 3/3 windows, two genuine seed replicates. NNNC: +1.02pp unresolved on W1 but **−1.71pp RESOLVED harm on W2**, degrading monotonically as the gate opens |
| `exit_rvol_threshold` | 2.0 *(dead code)* | **1.0** | The interior optimum 1.0 REPLICATES on both families despite their entry rvol differing 10x. Inert on NNNC while the signal is off |
| `cexit_take_profit` | **0.024** | **0.022** | NNMT's 0.022 vs 0.032 measured null (−0.258pp W1, inside a 0.53pp band) |
| `cexit_max_days` | **10** | **21** | Long-standing, never a variable |

## Two rules that follow from the measurements

**1. Do NOT open either family's guards.** This is measured on both, at scale:

| | opening the guards costs |
|---|---|
| NNMT | **−34.15pp** on W1 (2798 trades, −23.4%) |
| NNNC | real model returns **+4.35%** (2 seeds) against production's **+10.446%** |

On NNNC the open gate exposes **+17.824pp of headroom** and still loses 5–6pp of
realised return, because the model cannot use it. The guards are performing the
selection the model cannot. See `NNNC_PATH_AND_EXIT.md`.

**2. Guard effects are NOT additive — a leave-one-out cannot rank them.** On
NNMT the joint effect of removing all six is −34.150pp against a −8.123pp sum of
the individual effects: **4.20x super-additive**. The chain is load-bearing as a
conjunction; no single guard is. Never justify opening one guard from its own
leave-one-out number.

## WARNING — reading these values programmatically

freqtrade's `Parameter` objects are **class-level singletons on `BaseStrategy`**.
Loading two strategies in the SAME process contaminates the second: values the
first set persist wherever the second does not override them. Loading NNNC then
NNMT in one process reported NNMT's take-profit as 0.024/10 days when it is
actually 0.022/21.

**Always load one strategy per process** when dumping effective configuration,
and note that `ft_load_hyper_params()` runs inside `ft_bot_start()` — a bare
`StrategyResolver.load_strategy()` returns raw `DecimalParameter` DEFAULTS, not
the live merged values.
