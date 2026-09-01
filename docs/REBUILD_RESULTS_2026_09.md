# Rebuild results — strategy P&L on the fixed GANs (2026-09-01)

Backtests of every strategy against GANs rebuilt with the fidelity work from
2026-08-29/31 (covariance/correlation terms on five variants, the per-column
Gumbel fix, G-clip removal, MLX-matched generator).

**Window:** `test_strat.sh -n 720 -o 30` — 720 days ending 30 days ago, keeping a
30-day OOS tail. This is NOT the window behind the `+6.61% / 3.12% DD` figure in
`project_current_state_2026_05_26`, so these are a fresh baseline set, not a
like-for-like comparison with it.

**GAN fidelity for these exact models** is in GAN_SCORECARD.md
("PRODUCTION-MODEL FIDELITY"): five of six healthy on both axes
(sigma 1.00-1.07, dcorr 0.028-0.083); `mt_ddpm` over-dispersed (sigma 2.13).

## Results

| strategy | GAN | trades | avg % | tot % | drawdown | win% |
|---|---|---|---|---|---|---|
| NNNC_MLX | none (baseline) | 501 | 0.18 | **+3.98** | 7.26% | 57.7 |
| NNNC_DDPM_MLX | TabDDPM | 490 | 0.15 | **+4.34** | **6.64%** | 57.3 |
| NNNC_WGAN_MLX | WGAN-MLX | 512 | 0.09 | **+1.43** | 8.41% | 57.0 |
| NNNC_CGP_MLX | CTAB-GAN+ | 147 | 0.47 | **+4.34** | **2.92%** | **62.6** |
| NNMT_MLX | none (baseline) | 531 | 0.17 | **+7.90** | 5.77% | 53.9 |
| NNMT_DDPM_MLX | MT_DDPM | 524 | 0.14 | **+4.38** | 6.26% | 53.4 |
| NNMT_WGAN_MLX | MT_WGAN | 508 | 0.16 | **+5.29** | 5.69% | 54.1 |
| NNMT_CGP_MLX | MT_CTAB-GAN+ | 524 | 0.20 | **+7.29** | 6.27% | 54.2 |

## Reading so far — encouraging, NOT established

`NNNC_DDPM_MLX` beats the non-GAN baseline on **both** axes: +4.34% vs +3.98%
(+0.36pp) and 6.64% vs 7.26% drawdown. That is directionally against the standing
"non-GAN wins" verdict (GAN_TODO section 1, where non-GAN won 2/3 seeds).

**Do not over-read it.** It is one seed on one window, and the prior work records
GAN-vs-non-GAN gaps of ~1pp as within noise at this trade count
([[feedback_evaluate_interventions_at_powered_operating_point]]). A 0.36pp gap is
well inside that band. What would make it real: the paired-seed protocol at a
loosened operating point, which is the task-12 experiment.

What HAS changed since that verdict is the input: those runs used generators
emitting badly wrong joint structure, and these do not
(see the production-fidelity table). So the question is genuinely reopened, even
though this single pair of runs does not settle it.

## !! CONFOUND — only ONE arm has hyperopt parameters !!

**`NNNC_CGP_MLX` is the only strategy with a tuned parameter file.** Confirmed in
the logs: it loads `NNNC/NNNC_CGP_MLX.json` (and reports
`minimal_roi: {'0': 0.03}`); the other three load nothing and fall back to
`NNNCStrategy.buy_params`.

| param | CGP (tuned JSON) | the other three (class default) |
|---|---|---|
| `prediction_threshold` | **0.29** | **0.69** |
| `entry_adx_threshold` | 10.0 | 38.0 |
| `entry_rvol_threshold` | 1.9 | 0.5 |
| `entry_close_norm_threshold` | 0.6 | -0.1 |
| `entry_guard_threshold` | 0.1 | 0.3 |
| ROI / stoploss | 0.03 / -0.05 | class default |
| **trailing stop** | **enabled** (0.01 / 0.03) | not set |
| exit params | tuned (`cexit_take_profit` 0.04, `enable_exit_signal` true, ...) | class default |

CGP does not merely have different entry thresholds — it has an entirely
different RISK-MANAGEMENT configuration: trailing stop on, 3% ROI, -5% stoploss,
tuned exits, `max_open_trades` 10.

**Consequence: the CGP row measures hyperopt configuration at least as much as it
measures the GAN.** Its standout drawdown (2.92% vs 6.64%) is very plausibly the
trailing stop and ROI, not CTAB-GAN+. **The earlier "CTAB-GAN+ is the standout"
reading is withdrawn** — it is not a valid GAN comparison.

**What still IS valid:** `NNNC_MLX`, `NNNC_DDPM_MLX` and `NNNC_WGAN_MLX` all run
the SAME class defaults with no parameter file, so those three ARE comparable to
each other. The WGAN finding below survives; the CGP finding does not.

**Also supports a plot-level observation from the user:** NNNC_WGAN_MLX's entry
signals look good on inspection while its P&L is poor. It runs
`prediction_threshold` 0.69 and `entry_adx_threshold` 38 — parameters never tuned
against a WGAN-augmented model. Good signals gated by mismatched thresholds is
consistent with what the backtest shows.

**To get a real GAN comparison:** run every arm under IDENTICAL parameters (the
cheap option — ignore/rename the one JSON), or hyperopt each arm separately (the
expensive but fairer one). Until then, cross-arm P&L differences confound GAN
choice with parameter tuning.

## Fidelity does NOT predict P&L — in either direction

**CORRECTION.** With three arms this looked "close to inverse"; the fourth shows
it is simply UNRELATED. Fidelity of the GAN and P&L of its strategy are
independent, and WGAN is separately harmful.

| strategy | trades | tot % | drawdown | win% | profit/DD | GAN dcorr | GAN sigma |
|---|---|---|---|---|---|---|---|
| NNNC_CGP_MLX ⚠ | 147 | +4.34 | 2.92% | 62.6 | 1.49 | 0.083 (worst) | 1.038 |
| NNNC_DDPM_MLX | 490 | +4.34 | 6.64% | 57.3 | 0.65 | 0.028 (best) | 1.025 |
| NNNC_MLX (no GAN) | 501 | +3.98 | 7.26% | 57.7 | 0.55 | — | — |
| NNNC_WGAN_MLX | 512 | +1.43 | 8.41% | 57.0 | 0.17 | 0.064 | 1.061 |

The best-fidelity GAN (TabDDPM, dcorr 0.028) and the worst-fidelity GAN
(CTAB-GAN+, 0.083) produce the SAME total profit, while the middle one (WGAN,
0.064) is far worse than both. There is no monotone relationship in either
direction.

**Two things do hold:**

1. **WGAN augmentation is specifically harmful** — 2.55pp below the non-GAN
   baseline with a worse drawdown, replicating
   [[project_wgan_fidelity_fix_pnl_neutral]] (~3pp on 4/4 seeds). Improving its
   dcorr ~6x did not rescue it, so the penalty is not a fidelity defect.
2. ~~CTAB-GAN+ is the standout on risk-adjusted return~~ **WITHDRAWN** — see the
   confound above. CGP is the only arm with a tuned parameter file (trailing stop,
   ROI, stoploss, tuned entries and exits), so its numbers are not comparable to
   the other three.

CTAB-GAN+ is also the family this session's main work targeted (joint structure
0.043 -> 0.240 on the fixture; the production model measures dcorr 0.083,
sigma 1.038). That is suggestive but NOT attributable — there is no pre-fix
CTAB-GAN+ arm on this window to compare against.

**Caveat on CGP specifically: 147 trades vs ~500 for the others.** It is a much
more selective operating point, so its low drawdown is partly structural, and the
sample is ~3x smaller. Comparisons across arms with a 3x trade-count difference
are not like-for-like ([[feedback_evaluate_interventions_at_powered_operating_point]]).
The honest read is "promising and worth a powered re-test", not "CGP wins".

## Method note — run strategies ONE PER PROCESS

The first attempt ran all 17 rebuild steps as a single background process and was
SIGKILLed ~2h in, during `NNNC_DDPM_MLX`. This is the known failure recorded in
`project_noisycoconut_infra` ("LONG background batches get SIGKILLed — external,
not OOM; single runs survive"). `rebuild.sh` is exactly such a batch.

The GANs and scalers survived (all 7 complete); only the strategy loop was lost.
Remaining strategies are being run one process each via `/tmp/run_one_strat.sh`.

A second trap in the same attempt: `rebuild.sh` calls bare `freqtrade`, which
needs the venv on PATH. Run without it, every step failed instantly with
`command not found` while the script — which has no `set -e` — still exited 0,
AFTER its `rm -rf saved_data` had already succeeded. Back up `saved_data` before
running it.


## NNMT (multi-task) baseline roughly DOUBLES the NNNC (single-task) baseline

Both run class defaults with no parameter file, so this pair IS comparable:

| baseline | trades | tot % | drawdown | win% | profit/DD |
|---|---|---|---|---|---|
| **NNMT_MLX** (multi-task) | 531 | **+7.90** | **5.77%** | 53.9 | **1.37** |
| NNNC_MLX (single-task) | 501 | +3.98 | 7.26% | 57.7 | 0.55 |

Nearly 2x the profit at a LOWER drawdown, on a similar trade count — and NNMT
gets there with a LOWER win rate (53.9% vs 57.7%), so the edge is in payoff size,
not hit rate.

**Do not read this as "multi-task learning works".** NNMT and NNNC differ in more
than the auxiliary heads: different architecture, and critically a different
TRADING LABEL (NNMT's is built by filtering raw buy/sell signals through the
ground-truth task values — see NNMT_TASK_AUDIT.md). The label definition alone
could account for the gap. What it does establish is that the multi-task FAMILY
is the stronger starting point on this window, which raises the value of the
queued task-ablation runs.


## GAN augmentation vs its own baseline — both families, class defaults only

Comparing each GAN arm ONLY against the baseline in its own family, and only
where neither has a parameter file (so these are like-for-like):

| family | arm | tot % | delta vs baseline | drawdown | GAN fidelity |
|---|---|---|---|---|---|
| NNNC | baseline | +3.98 | — | 7.26% | — |
| NNNC | TabDDPM | +4.34 | **+0.36** | 6.64% | healthy (dcorr 0.028, sigma 1.025) |
| NNNC | WGAN | +1.43 | **-2.55** | 8.41% | healthy (dcorr 0.064, sigma 1.061) |
| NNMT | baseline | +7.90 | — | 5.77% | — |
| NNMT | MT_DDPM | +4.38 | **-3.52** | 6.26% | **DEFECTIVE (sigma 2.13, 13.6% clipped)** |

**Three of the four GAN arms so far are neutral-to-harmful**, and the one
positive delta (+0.36pp) sits inside the noise band this project documents at
this trade count.

### The fidelity-to-P&L link is still not clean

Tempting story: MT_DDPM is the one GAN with a fidelity DEFECT and it produces the
worst delta (-3.52pp), so bad fidelity causes bad P&L. **But WGAN-MLX has
EXCELLENT fidelity (dcorr 0.064, sigma 1.061) and is also strongly harmful
(-2.55pp).** So a fidelity defect is not necessary for a P&L penalty, and good
fidelity is not sufficient to avoid one.

The most defensible summary remains: **GAN augmentation on this pipeline ranges
from neutral to harmful, and fidelity does not predict which.** That is the
information-ceiling conclusion, now re-tested against generators far better than
the ones it was originally established on.


## The WGAN penalty REPLICATES across families — and it is not fidelity

| family | baseline | WGAN arm | delta | WGAN model fidelity |
|---|---|---|---|---|
| NNNC | +3.98 | +1.43 | **-2.55pp** | dcorr 0.064, sigma 1.061 |
| NNMT | +7.90 | +5.29 | **-2.61pp** | **dcorr 0.043, sigma 1.002** |

Two different families, two separately-trained WGAN models, and the penalty is
the same size to within 0.06pp. It also matches the ~3pp cost measured on 4/4
seeds in [[project_wgan_fidelity_fix_pnl_neutral]], which used the PRE-FIX
generators.

**MT_WGAN-MLX is the single best-fidelity model in the whole family**
(sigma 1.002, dcorr 0.043 — see the production-fidelity table) and it still costs
2.61pp. Three independent observations now, spanning generators from badly-broken
to near-perfect, all showing the same ~2.6-3pp WGAN penalty.

**Conclusion: the WGAN penalty is a property of WGAN augmentation itself, not of
synth quality.** Improving WGAN fidelity — which this session did substantially,
dcorr ~0.21 -> 0.043 on the MT variant — moved it not at all. Whatever WGAN synth
does to the classifier, it is not something better marginals or joints fix.

This is the strongest fidelity-vs-edge evidence in the codebase: same
intervention, three fidelity levels, one outcome.


---

# FINAL VERDICT — all 8 arms, task 12 answered

## The clean comparison: NNMT family, no parameter files anywhere

| arm | trades | tot % | delta vs baseline | drawdown | GAN fidelity |
|---|---|---|---|---|---|
| **NNMT_MLX (no GAN)** | 531 | **+7.90** | — | **5.77%** | — |
| NNMT_CGP_MLX | 524 | +7.29 | **-0.61** | 6.27% | dcorr n/a, sigma 1.073 |
| NNMT_WGAN_MLX | 508 | +5.29 | **-2.61** | 5.69% | dcorr 0.043, sigma 1.002 |
| NNMT_DDPM_MLX | 524 | +4.38 | **-3.52** | 6.26% | sigma 2.13 (defective) |

**No GAN arm beats the baseline. The best is neutral, the rest are harmful.**
All four run identical class defaults, so this is a clean four-way test.

## NNNC family (CGP row confounded — it is the only arm with a tuned JSON)

| arm | trades | tot % | delta | drawdown |
|---|---|---|---|---|
| NNNC_MLX (no GAN) | 501 | +3.98 | — | 7.26% |
| NNNC_DDPM_MLX | 490 | +4.34 | +0.36 | 6.64% |
| NNNC_WGAN_MLX | 512 | +1.43 | -2.55 | 8.41% |
| NNNC_CGP_MLX (tuned JSON) | 147 | +4.34 | n/c | 2.92% |

The single positive delta in the whole study (+0.36pp) sits inside the documented
noise band at this trade count.

## Task 12 answered: working synth does NOT move P&L

The rebuild's purpose was to re-test "GAN augmentation does not help at any
fidelity" using generators that actually work. Five of six GANs are now healthy
on BOTH axes on real production data (sigma 1.00-1.07, dcorr 0.028-0.083 — see
GAN_SCORECARD.md). **The verdict holds.** Across 6 like-for-like GAN-vs-baseline
comparisons, the best outcome is neutral.

**The WGAN result is the sharpest evidence.** The penalty is -2.55pp (NNNC) and
-2.61pp (NNMT) — identical to within 0.06pp across two families and two
separately-trained models — and it matches the ~3pp measured on 4/4 seeds with
the PRE-FIX generators. MT_WGAN-MLX is the best-fidelity model in the codebase
(sigma 1.002, dcorr 0.043) and still costs 2.61pp. Same intervention, three
fidelity levels spanning broken to near-perfect, one outcome.

## The bigger finding is not about GANs

**The multi-task baseline nearly doubles the single-task baseline**: +7.90% vs
+3.98% at a lower drawdown (5.77% vs 7.26%), on a comparable trade count, with a
LOWER win rate — so the edge is payoff size, not hit rate. That gap dwarfs every
GAN effect measured here. If there is upside on this pipeline it is in the
NNMT direction, not in synth quality.

## What this does NOT establish

- **Parameters were never tuned per arm.** Every arm except NNNC_CGP runs
  `prediction_threshold` 0.69; the one tuned config uses **0.29**. A user
  inspection found NNNC_WGAN's entry signals look good on a plot despite poor
  P&L. So "harmful" here means **harmful AT THESE PARAMETERS** — a per-arm
  hyperopt could change the ordering, and is the obvious next experiment.
- Single seed, single window, no paired-seed protocol.
- MT_DDPM's fidelity defect (sigma 2.13) is a real confound for that one row.
