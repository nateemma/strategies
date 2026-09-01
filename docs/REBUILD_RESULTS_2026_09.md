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
| NNMT_MLX | none (baseline) | _pending_ | | | | |
| NNMT_DDPM_MLX | MT_DDPM | _pending_ | | | | |
| NNMT_WGAN_MLX | MT_WGAN | _pending_ | | | | |
| NNMT_CGP_MLX | MT_CTAB-GAN+ | _pending_ | | | | |

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

## Fidelity does NOT predict P&L — in either direction

**CORRECTION.** With three arms this looked "close to inverse"; the fourth shows
it is simply UNRELATED. Fidelity of the GAN and P&L of its strategy are
independent, and WGAN is separately harmful.

| strategy | trades | tot % | drawdown | win% | profit/DD | GAN dcorr | GAN sigma |
|---|---|---|---|---|---|---|---|
| **NNNC_CGP_MLX** | 147 | **+4.34** | **2.92%** | **62.6** | **1.49** | 0.083 (worst) | 1.038 |
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
2. **CTAB-GAN+ is the standout on risk-adjusted return** — same profit as DDPM at
   44% of the drawdown and a 5pp higher win rate. profit/DD 1.49 vs 0.65.

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
