# 040 NNMT parity & head study -- report

## 1. Baseline: NNMT_MLX as it stands (`mt`, seed 1, T012)

15/15 folds recorded (`regime/arms/mt_fNN_FNN.json`), each with a trained model and trades.
Key a97a4b9cdde4b5a5 (f00, after the T014 key hygiene); trained BEFORE the framework port.

| arm | early f00-f07 | late f08-f14 | total | trades |
|---|---|---|---|---|
| mt | +12.43 | +8.57 | +21.00 | 795 |
| cs_aro | +40.70 | +19.25 | +59.95 | 930 |
| w_aro | +35.85 | +15.59 | +51.44 | 930 |
| st_g24_rev | | | 71.5 | |

Per fold, wallet profit % (trades):

| fold | mt | cs_aro | w_aro |
|---|---|---|---|
| f00 | +1.07 (40) | +4.27 (27) | +3.76 (27) |
| f01 | -3.91 (26) | +3.96 (25) | +3.72 (25) |
| f02 | +0.98 (22) | +1.24 (10) | +1.11 (10) |
| f03 | +2.77 (44) | +1.03 (16) | +0.79 (16) |
| f04 | +3.61 (98) | +11.95 (128) | +10.78 (128) |
| f05 | +1.42 (56) | +0.89 (60) | +0.89 (60) |
| f06 | -1.02 (44) | +3.77 (46) | +3.19 (46) |
| f07 | +7.50 (115) | +13.59 (99) | +11.60 (99) |
| f08 | -0.71 (144) | +3.39 (147) | +2.63 (147) |
| f09 | +0.65 (19) | +2.65 (39) | +2.70 (39) |
| f10 | -0.56 (17) | -0.01 (3) | +0.06 (3) |
| f11 | +4.41 (96) | +6.38 (208) | +3.73 (208) |
| f12 | +1.93 (42) | +2.54 (39) | +2.48 (39) |
| f13 | +2.29 (24) | +2.43 (37) | +1.67 (37) |
| f14 | +0.56 (8) | +1.86 (46) | +2.32 (46) |

Descriptive only (no rule is applied to the baseline): NNMT trails cs_aro by ~39 pp, in both halves;
most of the gap is f04/f07/f08, where all three books trade most.

## 2. Framework port is inert (T017-T018)

`mt_fw0` (NNMT with EntryFrameworkMixin, every switch off, same model key -> same weights) is
trade-identical to `mt` on 15/15 folds (`trades_hash`; `verdicts_models.json["identity"]`). Run with
`FT_STOP_INVARIANTS=report` and a sink: 0 stop-invariant violations on all 15 folds.

## 3. Operating point (T021-T022)

Model-decided share (1 - entries / guard-passing bars): `mt` **0.001** -- today's NNMT entries are
decided by the guard chain, not the model -- vs `mt_op` (guards open, threshold 0.80) **0.841** (>= 0.5:
model and head arms at the operating point are readable).

| arm | early | late | total | trades | verdict vs mt_fw0 |
|---|---|---|---|---|---|
| mt_fw0 | +12.43 | +8.57 | +21.00 | 795 | -- |
| mt_op | +27.38 | -18.83 | +8.55 | 10,648 | **FAIL** (late gain -27.40, INCONCLUSIVE_ARM_LOSES p 0.02; maxDD 4.5 -> 32.1 % WORSENED) |

Beside the comparators, `mt` FAILs vs cs_aro, w_aro, wf_aro_g24 and st_g24_rev (all WORSENED).
The R7 selection is re-run after the variants (T028); `mt_op` is its reference.

## §4 Architecture amendment: shared LSTM + per-task LSTM (`mt_ls`), screen

Pre-registration `preregistration_arch.json` (b5b640a6); verdict `verdicts_arch.json`. Early half f00–f07, at the
operating point, vs `mt_lr` (shared LSTM + MLP per task).

| arm | early pp | trades | model share |
|---|---|---|---|
| `mt_ls` | 26.02 | 5,696 | 0.85 |
| `mt_lr` (control) | 27.22 | 5,861 | — |

**Screen: FAIL** (early gain −1.19 pp; paired UNCHANGED, Δ −1.24, CI [−6.9, +5.5] pp/yr, p 0.84; risk UNCHANGED).
By rule there is no confirmation stage and no seed 2. Per-task LSTMs neither help nor clearly hurt; together with
`mt_lr` ≈ `mt_op`, the network design inside the multi-task heads moves the early half by about a point.

## §5 Model variants vs `mt_op` and the R7 selection (T028, seed 1)

Pre-registration `preregistration_models.json`; verdicts `verdicts_models.json` (`judge models`, 2026-10-22).
All arms at the operating point, 15 folds, model share 0.84 for every arm.

| arm | total pp | early gain vs `mt_op` | late gain | paired | decision |
|---|---|---|---|---|---|
| `mt_op` (control) | 8.55 | — | — | — | (FAIL vs `mt_fw0`, §3) |
| `mt_lr` plain LSTM | 8.29 | −0.16 | −0.10 | UNCHANGED p 0.97 | FAIL |
| `mt_lr_ri` plain LSTM, raw + 24 indicators | 9.48 | +0.25 | +0.68 | UNCHANGED p 0.89 | **PASS, seed 2 required (gain < 1 pp)** |
| `mt_raw` raw inputs, original network | −11.77 | −20.78 | +0.46 | — | FAIL |
| `mt_lr_raw` plain LSTM, raw only | −18.56 | −18.33 | −8.78 | — | FAIL |

**R7 selection: `mt_lr_ri`** (highest pooled pp among arms with no WORSENED half). Its margin is noise-sized
(+0.93 pp over 15 folds, CI [−3.5, +4.1] pp/yr; Sharpe and maxDD UNCHANGED), which is what the seed-2 rule is
for: seed 2 of `mt_lr_ri` and `mt_op` started 2026-10-22 01:23.

**Seed 2 (finished 2026-10-22 21:18; `verdicts_models.json` → `seed2`): selection CONFIRMED.**

| seed | `mt_lr_ri` | `mt_op` | gain (early / late) | paired | Sharpe / maxDD |
|---|---|---|---|---|---|
| 1 | 9.48 | 8.55 | +0.93 (+0.25 / +0.68) | UNCHANGED, CI [−3.5, +4.1], p 0.89 | UNCHANGED |
| 2 | 13.75 | 10.04 | +3.71 (+3.08 / +0.63) | UNCHANGED, CI [−2.7, +4.6], p 0.60 | UNCHANGED (0.31 vs 0.23; 32.0 vs 33.5 %) |

`mt_lr_ri` beats `mt_op` in both halves on both seeds; seed 2's gain clears the 1 pp seed rule, so it PASSES
and stays `<M>`. The difference is not significant on either seed (paired UNCHANGED): "no worse, consistently
slightly better", not a demonstrated improvement. `mt_op` seed 2 vs `mt_fw0` FAILs again (late −28.2 pp, risk),
so the late loss of the operating point replicates.

**Reading:** with indicators in the input, the network barely matters (three designs within 1.2 pp). Raw-only
inputs lose ~19-21 pp early. Every variant keeps the operating point's late loss (≈ −18 pp), so the late
failure in §3 belongs to the operating point, not the network.

## §6 Regime transition (T042–T046)

**T042 (2026-10-23): labels built, causality PASS.** 1,800,791 rows (10 screen pairs). Truncation test: z/state
288 comparisons and label 288 comparisons all equal on the overlap; every one of the 32 stand-in inputs has a
PASS in 037's truncation tables (`screens_transition.json` → `truncation`).

Base rates (mean per-fold share of tradeable test bars; events = run starts, summed per half):

| cell | early rate | late rate | events early / late |
|---|---|---|---|
| up_pair_96 | 0.105 | 0.100 | 6,504 / 5,123 |
| down_pair_96 | 0.098 | 0.108 | 7,053 / 5,913 |
| up_btc_96 | 0.127 | 0.107 | 7,297 / 5,689 |
| down_btc_96 | 0.105 | 0.107 | 6,474 / 5,997 |
| up_pair_288 | 0.109 | 0.100 | 3,962 / 2,816 |
| down_pair_288 | 0.101 | 0.117 | 3,639 / 3,496 |
| up_btc_288 | 0.140 | 0.099 | 4,290 / 4,135 |
| down_btc_288 | 0.108 | 0.121 | 3,541 / 3,621 |
| up_pair_672 | 0.116 | 0.099 | 2,810 / 1,902 |
| down_pair_672 | 0.107 | 0.118 | 2,427 / 2,603 |
| up_btc_672 | 0.170 | 0.106 | 2,826 / 1,877 |
| down_btc_672 | 0.115 | 0.112 | 2,796 / 2,723 |

Every cell clears the 200-events-per-half floor by a wide margin.

**Transition screen (T043, `screens_transition.json`): PASS, cells selected `up_pair_96` (UP) and `down_pair_672`
(DOWN).** Every cell ranks transitions above the trend baseline in both halves (stand-in AUC 0.56–0.65 vs trend
0.37–0.47); cells that fail do so on the tradeability proxy (top-quintile forward return vs all bars), recorded per
cell as "NOT TRADEABLE AT THIS HORIZON" in `verdicts_transition.json` → `cells`.

**Arms (T044, stand-in predictions on `cs_aro`'s book; `verdicts_transition.json`):**

| arm | use | early | late | total | trades | verdict |
|---|---|---|---|---|---|---|
| `cs_aro` (control) | — | +40.70 | +19.25 | +59.95 | 930 | — |
| `tr_scale` | stake ×1.33 at P(UP) ≥ tail p80, ×0.67 at P(DOWN) ≥ p80 | +42.61 | +22.74 | +65.35 | 930 | **PASS (both seeds)**; paired IMPROVED p 0.02, Sharpe/maxDD UNCHANGED |
| `tr_scale` seed 2 (bootstrap stand-in, decisive) | | +43.40 | +23.56 | +66.97 | 930 | PASS (+2.70 / +4.32) |
| `tr_gate` | skip entry at P(DOWN) ≥ p80 | +38.31 | +17.26 | +55.58 | 896 | FAIL (both halves negative) |
| `tr_exit` | close at P(DOWN) ≥ p90 | +31.79 | +11.96 | +43.75 | 1,013 | FAIL (both halves, paired, risk) |
| `tr_long` | standalone long at P(UP) ≥ p99 (st_g24_rev exits) | +21.39 | +10.64 | +32.03 | 977 | **PASS sign-vs-zero (both seeds)**; per-trade CI [0.36, 0.91] pp |
| `tr_long` seed 2 (bootstrap) | | +18.99 | +14.67 | +33.65 | 1,148 | PASS |

`tr_scale` trades `cs_aro`'s exact trades (same exits); its gain is sizing only, and about half of it is extra
capital deployed (DESCRIPTIVE). `tr_long`'s comparator `st_g24_rev` (63.56 / 7.94, 2,518 trades) is reported, not
judged.

**NN head (T045): NO MATCH.** NNMT heads trained on the selected cells (`MT_LR_RI_TRH96`, `MT_LR_RI_TRH672`) fall
short of the stand-in beyond the 0.01 tolerance (up_pair_96 −0.037 early / −0.019 late; down_pair_672 −0.010 / −0.024),
so the arms are not re-run on NN tables: the transition signal is usable from the stand-in, not as an NNMT head.

## §7 Regime state (T056–T059, `verdicts_regime_state.json`, DESCRIPTIVE)

No regime-state cell passed its screen (`screens_regime_state.json` pass false), so `rs_gate` / `rs_scale` were not
run (recorded results). On the fallback cell `pair_288` the NN regime heads do not beat the baselines: `MT_LR_RI_RG`
BULL AUC 0.538 / 0.513 vs trailing-z 0.538 / 0.516; BEAR 0.522 / 0.513 (margin +0.02 / +0.01, below the bar).
Shared vs separate regime training: no difference. `mt_lr_ri_rgm` (the regime head trained) vs `mt_lr_ri`:
8.87 vs 9.48 pp (early +0.44, late −1.06), DESCRIPTIVE only.

## §8 Framework arms on the selected model (T030–T031, `verdicts_framework.json`)

All vs `mt_lr_ri` (operating point, same weights; TR has its own key).

| arm | early | late | total | trades | gain early / late | decision |
|---|---|---|---|---|---|---|
| `mt_lr_ri` (control) | +27.63 | −18.15 | +9.48 | 10,813 | — | — |
| `mt_lr_ri_g24` (24 h stop-loss grace) | +43.22 | −2.35 | +40.86 | 6,855 | +15.59 / +15.80 | **PASS** (paired UNCHANGED p 0.07; Sharpe IMPROVED, maxDD 30.5 → 19.3 % UNCHANGED) |
| `mt_lr_ri_g24` seed 2 vs `mt_lr_ri` seed 2 | +48.25 | −6.02 | +42.23 | 6,744 | +15.51 / +12.96 | **PASS (both seeds)** (control 13.75; Sharpe IMPROVED) |
| `mt_lr_ri_hpa` (in-profit held-pair add) | +26.32 | −24.62 | +1.70 | 10,477 | −1.31 / −6.47 | FAIL |
| `mt_lr_ri_gate` (gbb trigger gate) | +25.95 | −18.18 | +7.77 | 10,807 | −1.68 / −0.03 | FAIL (early risk WORSENED) |
| `mt_lr_ri_tp` (C2 ATR take-profit) | +21.57 | −11.62 | +9.95 | 10,958 | −6.06 / +6.53 | FAIL (early) |
| `mt_lr_ri_tr` (trading head only, twin) | +24.37 | −5.05 | +19.32 | 7,964 | −3.25 / +13.09 | FAIL (early) |

The gbb gate is near-inert on NNMT: the model's buy signals sit almost entirely on gbb-trigger bars (f05: 14
refused of 12,830). **Twin: no evidence that multi-task training helps** (the trading-head-only twin is +9.8 pp
better in total but fails the early half, so no evidence it hurts at the pre-registered bar either).

## §9 Auxiliary heads (T032–T033, `screens_heads.json`)

Each head beats its persistence / trailing baseline on held-out folds (skill PASS, all five), but none ranks the
width book's trades (relevance FAIL, all five), so no head arm was run (`verdicts_screens.json`: "not run").

| head (column) | AUC early / late | baseline AUC | relevance gap at share 0.5 | sign folds | pass |
|---|---|---|---|---|---|
| regime (`p_regime_bull`) | 0.830 / 0.827 | 0.789 / 0.778 | +0.006 | 5 / 8 | FAIL |
| risk (`p_risk_high`, −) | 0.935 / 0.932 | 0.705 / 0.699 | +0.001 | 6 / 8 | FAIL |
| momentum (`p_momentum_up`) | 0.962 / 0.964 | 0.796 / 0.808 | +0.000 | 5 / 8 | FAIL |
| flow (`p_flow_up`) | 0.907 / 0.910 | 0.690 / 0.696 | −0.039 | 7 / 8 (wrong sign) | FAIL |
| profit (`p_profit_profit`) | 0.758 / 0.754 | 0.549 / 0.536 | +0.009 | 5 / 8 | FAIL |
| *trading head `p_buy` (sizer screen)* | | | **+0.056** | 7 / 8 | PASS |

Instrument: 412 width-book trades f00–f07, control-at-share gap 0.100. The heads predict their own targets well,
but those targets carry no information about which width-book trades win.

## §10 Sizer: `p_buy` on `w_aro`'s width book (T034, `verdicts_screens.json`)

Identity twin `mt_lr_ri_wb` = `w_aro` on 15/15 folds (re-run after declaring `MODEL_ENTRY_FAMILY = "reversion"`,
which the sizer needs to tag the book's entries). Thresholds: per fold at the 0.5 agree share (achieved
0.494–0.506 both seeds; `sizer_thresholds{,_s2}.json`).

| arm | early | late | total | trades | gain vs `w_aro` early / late | decision |
|---|---|---|---|---|---|---|
| `w_aro` (control) | +35.85 | +15.59 | +51.44 | 930 | — | — |
| `mt_lr_ri_cs` seed 1 | +37.06 | +22.22 | +59.28 | 930 | +1.20 / +6.63 | PASS (paired IMPROVED p ≈ 0.001; Sharpe/maxDD UNCHANGED) |
| `mt_lr_ri_cs` seed 2 | +35.58 | +18.86 | +54.43 | 930 | −0.28 / +3.27 | FAIL (early gain; paired UNCHANGED p 0.25) |

**Decision: FAIL at seed 2.** Reported vs `cs_aro` (59.95) at seed 1: −3.65 / +2.97, FAIL. Both seeds trade
`w_aro`'s exact 930 trades (sizing only). The NNMT trading head sizes the width book about as well as ARO does
at seed 1 but does not replicate.

## §11 Every arm, per half (seed 1 unless marked)

Wallet profit %, summed per half (early f00–f07, late f08–f14). Δ = total minus the reference total
(DESCRIPTIVE for every column except the arm's own comparator, whose verdict is in the section cited). Model-decided
share is recorded only for model arms (`verdicts_models.json`); width-book, sizer and transition arms enter
model-free or on `cs_aro`'s book.

| arm | § | early | late | total | trades | share | own comparator → decision | Δ cs_aro | Δ w_aro | Δ st_g24_rev | Δ wf_aro_g24 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `cs_aro` | ref | +40.70 | +19.25 | +59.95 | 930 | — | — | 0 | +8.51 | −11.55 | +9.99 |
| `w_aro` | ref | +35.85 | +15.59 | +51.44 | 930 | — | — | −8.51 | 0 | −20.06 | +1.48 |
| `st_g24_rev` | ref | +63.56 | +7.94 | +71.50 | 2,518 | — | — | +11.55 | +20.06 | 0 | +21.54 |
| `wf_aro_g24` | ref | +35.28 | +14.68 | +49.96 | 1,176 | — | — | −9.99 | −1.48 | −21.54 | 0 |
| `mt` | 1 | +12.43 | +8.57 | +21.00 | 795 | 0.001 | — (baseline) | −38.95 | −30.44 | −50.50 | −28.96 |
| `mt_fw0` | 2 | +12.43 | +8.57 | +21.00 | 795 | — | `mt` → identical 15/15 | −38.95 | −30.44 | −50.50 | −28.96 |
| `mt_op` | 3 | +27.38 | −18.83 | +8.55 | 10,648 | 0.841 | `mt_fw0` → FAIL | −51.40 | −42.89 | −62.95 | −41.41 |
| `mt_lr` | 5 | +27.22 | −18.93 | +8.29 | 10,802 | 0.837 | `mt_op` → FAIL | −51.66 | −43.15 | −63.21 | −41.67 |
| `mt_lr_ri` | 5 | +27.63 | −18.15 | +9.48 | 10,813 | 0.837 | `mt_op` → PASS (both seeds), selected | −50.47 | −41.96 | −62.02 | −40.48 |
| `mt_lr_ri` s2 | 5 | +32.74 | −18.98 | +13.75 | 10,623 | — | `mt_op` s2 (10.04) → PASS | −46.20 | −37.69 | −57.75 | −36.21 |
| `mt_raw` | 5 | +6.60 | −18.37 | −11.77 | 10,869 | 0.840 | `mt_op` → FAIL | −71.72 | −63.21 | −83.27 | −61.73 |
| `mt_lr_raw` | 5 | +9.05 | −27.61 | −18.56 | 10,888 | 0.838 | `mt_op` → FAIL | −78.51 | −70.00 | −90.06 | −68.52 |
| `mt_ls` (early only) | 4 | +26.02 | — | — | 5,696 | 0.85 | `mt_lr` → screen FAIL | | | | |
| `mt_lr_ri_g24` | 8 | +43.22 | −2.35 | +40.86 | 6,855 | — | `mt_lr_ri` → PASS (both seeds) | −19.09 | −10.58 | −30.64 | −9.10 |
| `mt_lr_ri_g24` s2 | 8 | +48.25 | −6.02 | +42.23 | 6,744 | — | `mt_lr_ri` s2 → PASS | −17.72 | −9.21 | −29.27 | −7.73 |
| `mt_lr_ri_hpa` | 8 | +26.32 | −24.62 | +1.70 | 10,477 | — | `mt_lr_ri` → FAIL | −58.25 | −49.74 | −69.80 | −48.26 |
| `mt_lr_ri_gate` | 8 | +25.95 | −18.18 | +7.77 | 10,807 | — | `mt_lr_ri` → FAIL | −52.18 | −43.67 | −63.73 | −42.19 |
| `mt_lr_ri_tp` | 8 | +21.57 | −11.62 | +9.95 | 10,958 | — | `mt_lr_ri` → FAIL | −50.00 | −41.49 | −61.55 | −40.01 |
| `mt_lr_ri_tr` | 8 | +24.37 | −5.05 | +19.32 | 7,964 | — | `mt_lr_ri` → FAIL | −40.63 | −32.12 | −52.18 | −30.64 |
| `mt_lr_ri_rgm` | 7 | +28.07 | −19.20 | +8.87 | 10,949 | — | `mt_lr_ri` → DESCRIPTIVE | −51.08 | −42.57 | −62.63 | −41.09 |
| `mt_lr_ri_wb` | 10 | +35.85 | +15.59 | +51.44 | 930 | — | `w_aro` → identical 15/15 | −8.51 | 0 | −20.06 | +1.48 |
| `mt_lr_ri_cs` | 10 | +37.06 | +22.22 | +59.28 | 930 | — | `w_aro` → FAIL at seed 2 | −0.67 | +7.84 | −12.22 | +9.32 |
| `mt_lr_ri_cs` s2 | 10 | +35.58 | +18.86 | +54.43 | 930 | — | `w_aro` → FAIL | −5.52 | +2.99 | −17.07 | +4.47 |
| `tr_scale` | 6 | +42.61 | +22.74 | +65.35 | 930 | — | `cs_aro` → PASS (both seeds) | +5.40 | +13.91 | −6.15 | +15.39 |
| `tr_scale` s2 (bs1) | 6 | +43.40 | +23.56 | +66.97 | 930 | — | `cs_aro` → PASS | +7.02 | +15.53 | −4.53 | +17.01 |
| `tr_gate` | 6 | +38.31 | +17.26 | +55.58 | 896 | — | `cs_aro` → FAIL | −4.37 | +4.14 | −15.92 | +5.62 |
| `tr_exit` | 6 | +31.79 | +11.96 | +43.75 | 1,013 | — | `cs_aro` → FAIL | −16.20 | −7.69 | −27.75 | −6.21 |
| `tr_long` | 6 | +21.39 | +10.64 | +32.03 | 977 | — | 0 (sign) → PASS (both seeds) | −27.92 | −19.41 | −39.47 | −17.93 |
| `tr_long` s2 (bs1) | 6 | +18.99 | +14.67 | +33.65 | 1,148 | — | 0 (sign) → PASS | −26.30 | −17.79 | −37.85 | −16.31 |

## §12 Verdicts (FR-015) and candidate defaults (T048)

All PROVISIONAL: one era (2023–26), halves as the persistence test. **Nothing has been adopted; no NNMT_MLX or
NNMTStrategy default was changed beyond the inert framework port. Adoption is the user's call.**

**1. NNMT defaults.** Candidates, in the order the evidence supports them:
- **24 h stop-loss grace (`mt_lr_ri_g24`)**: the only framework switch that passes, on both seeds (+15.6 / +15.8 and
  +15.5 / +13.0 pp per half vs `mt_lr_ri`; Sharpe IMPROVED). Candidate NNMT default.
- **`mt_lr_ri` as the NNMT base model** (plain LSTM on raw + 24 indicators): selected by R7 and confirmed at seed 2,
  but only "no worse, consistently slightly better" than `mt_op` (paired UNCHANGED both seeds).
- **Not candidates:** the operating point alone (late −27 pp vs `mt_fw0`, both seeds), the in-profit held-pair add,
  the gbb gate (near-inert), the C2 take-profit, and the `p_buy` sizer (fails at seed 2).
- Context (DESCRIPTIVE): even with g24, NNMT at the operating point trails every NNNC reference (−18 to −19 pp vs
  `cs_aro`, −9 to −11 vs `w_aro`) and its late half stays negative (−2.35 / −6.02). The g24 arm was judged against
  `mt_lr_ri`, not against today's guard-on `mt` (21.00).

**2. Auxiliary heads: don't use any of them.** Regime, risk, momentum, flow and profit each predict their own target
beyond a persistence baseline, but none ranks width-book trades (relevance FAIL), so none is used as a gate,
sizer or exit; no arm was run. The trading head's `p_buy` does rank them (sizer screen PASS) but the sizer arm
fails at seed 2. Multi-task training shows no measurable benefit (the trading-head-only twin is not worse at the
bar, and is better in total). The regime-state heads do not beat trailing-z / Markov baselines (§7).

**3. Regime transition: use the stand-in signal as an exposure scaler on `cs_aro` (candidate); not as an NNMT
head.** `tr_scale` passes on both seeds (+5.4 / +7.0 pp total vs `cs_aro`, both halves positive; sizing only,
about half of it extra capital). `tr_long` is a positive standalone sleeve on both seeds (32.0 / 33.7 pp) but
far smaller than `st_g24_rev`. Gating entries or exiting on P(DOWN) loses (`tr_gate`, `tr_exit` FAIL). An NNMT
head trained on the same target does not reach the stand-in (T045), so the transition signal lives outside NNMT.
