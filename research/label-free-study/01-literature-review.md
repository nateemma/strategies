# 01 — Literature review: label-free and objective-driven trading models

Compiled 2026-09-10 from four independent surveys (≈130 searches/fetches). Evidence grades:
**strong** = multi-year, multi-asset, net of costs, replicated or primary-source;
**medium** = credible but narrow (one group, one asset class, costs partly handled);
**weak** = short windows, no costs, single seed, unrefereed; **anecdotal** = blog / single
practitioner. Where a line is our inference rather than a source's claim it is marked *(inference)*.

---

## A. Direct policy optimisation — differentiable P&L objectives

The oldest and best-evidenced label-free family. A network outputs a position; the loss is a
function of realised returns net of costs; gradients flow through the P&L.

| Work | Setup | Result | Grade |
|---|---|---|---|
| Moody & Saffell, *Learning to Trade via Direct Reinforcement*, IEEE TNN 2001 ([link](https://ieeexplore.ieee.org/document/935097/)) | RRL: `F_t = tanh(w·x_t + u·F_{t−1})`, cost `δ|F_t−F_{t−1}|` in the return, trained on the **differential Sharpe ratio** | Beat Q-learning on S&P/T-bill; USD/GBP 30-min Sharpe 2.3; degrades predictably at 0.2–1% costs | medium (old, short tests) |
| Deng et al., *Deep Direct RL for Financial Signal Representation*, TNNLS 2016 ([link](https://pubmed.ncbi.nlm.nih.gov/26890927/)) | Autoencoder + k-means **unsupervised pretraining, then profit fine-tuning** ("task-aware BPTT"); minute futures | Positive on 3 Chinese futures; behaves like trend-following; weak in low vol | weak-medium |
| Lim, Zohren & Roberts, *Deep Momentum Networks*, 2019 ([arXiv](https://arxiv.org/abs/1904.04912)) | 88 futures; Sharpe vs MSE vs BCE losses; **turnover regulariser** | Sharpe loss >2× classical TSMOM before costs; edge survives only to **2–3 bp** | medium-strong |
| Zhang, Zohren & Roberts, *Deep Learning for Portfolio Optimisation*, 2020 ([arXiv](https://arxiv.org/abs/2005.13665)) | Single LSTM, **softmax long-only weights**, Sharpe loss, 4 ETFs | OOS Sharpe 1.96 vs 1.58 best fixed mix at 1 bp; gap collapses at 10 bp | weak-medium (4 assets) |
| Wood et al., *Momentum Transformer*, 2021 ([arXiv](https://arxiv.org/pdf/2112.08534v1), [code](https://github.com/kieranjwood/trading-momentum-transformer)) | 50 futures, 1995–2020 | Sharpe 2.54 vs 1.70 LSTM; **−0.35 at 3 bp** because costs were not in the loss | medium |
| Wood et al., *X-Trend*, 2023 ([arXiv](https://arxiv.org/abs/2310.10500)) | Few-shot cross-attention over a context of past regimes | +18.9% Sharpe vs neural forecaster 2018–23; 2× faster COVID recovery | medium |
| Ong & Herremans, 2023 ([arXiv](https://arxiv.org/abs/2306.13661)) | Sharpe head + auxiliary **volatility-forecast heads** | Improves Sharpe-trained portfolio up to 3 bp | medium |
| Wood, Roberts & Zohren, *DeePM*, Jan 2026 ([arXiv](https://arxiv.org/html/2601.05975v1), [code](https://github.com/kieranjwood/deepm)) | Pooled net Sharpe + **SoftMin worst-window Sharpe** (EVaR proxy); cost scale γ; seed ensembles; 2010–2025 net | ≈2× classical trend, +50% vs Momentum Transformer. Worst-window term "single largest driver of stability". **Training at γ=0.5 of true cost improved OOS net performance** | medium-strong (preprint) |
| Kubo & Nakagawa 2025 ([SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5020148)) | **Utility loss** μ − (λ/2)σ² vs Sharpe loss | Sharpe minibatch gradients are biased; utility loss unbiased and did better | weak-medium |
| Imaki et al., *No-Transaction Band Network*, 2021 ([arXiv](https://arxiv.org/pdf/2103.01775), [code](https://github.com/pfnet-research/NoTransactionBandNetwork)) | Output a band `[lo, hi]`, position = `clamp(prev, lo, hi)` | Optimal under proportional costs; converged in <100 sims where a plain net didn't in 1000 | strong (hedging) |
| Jansen, ML4Trading independent re-implementation ([link](https://insights.ml4trading.io/p/deep-learning-for-end-to-end-portfolio)) | Plain Sharpe-LSTM vs structured DeePM | **Naive Sharpe-LSTM: val Sharpe 1.61 → test 0.48**, below equal-weight. "Directly optimizing a portfolio loss does not remove overfitting." Only DeePM held | medium (practitioner) |
| Borrageiro et al., *RRL Crypto Agent*, IEEE Access 2022 ([PDF](https://discovery.ucl.ac.uk/id/eprint/10147117/1/The_Recurrent_Reinforcement_Learning_Crypto_Agent.pdf)) | 5-min BitMEX perp, utility loss, echo-state features | 350%/5yr net, IR 1.46 — but **71% of profit was funding** (unavailable on spot); costs ate 54% of gross; **IR 0.22–1.76 across 250 seeds** | medium |
| Jiang et al. 2017 ([ar5iv](https://ar5iv.labs.arxiv.org/html/1706.10059)) | 30-min Poloniex, log-return reward, zero slippage | "4× in 50 days"; **public replication flat** ([repo](https://github.com/akaniklaus/rl-portfolio-management)) | weak |
| Hourly BTC walk-forward under costs, 2026 ([arXiv](https://arxiv.org/abs/2606.00060)) | 27 folds 2018–25, supervised XGBoost / LSTM / iTransformer | Naive sign strategies collapse at 10 bp; a **cost-aware execution filter** (trade only if forecast > λ·c·|Δpos|) restores profit; XGBoost ≥ deep nets | medium |

**Takeaways for us.** (1) The mechanism is sound and maps directly onto the repo's
"learnability ≠ tradeability" finding. (2) Everything that worked had costs *in* the loss, a
turnover control, small models and seed ensembles. (3) Nothing has been validated at 15-min,
long-only, liquidity-capped spot — that is the gap brief 1 tests. (4) At ~10 bp/side the
holding period must be long relative to the bar; a no-trade band is the most transferable
trick. (5) Differentiable backtesters do not exist as libraries (`jaxbt` is a stub); the
Oxford-Man repos are the reference implementation; deep-hedging (`pfhedge`) is the mature
"backprop through P&L" analogue.

## B. Decision-focused / predict-then-optimise learning

Train a forecaster *through* the downstream optimiser (Donti, Amos & Kolter 2017; Elmachtoub &
Grigas SPO+; `cvxpylayers`, which supports MLX but has no Apple-GPU solver).

- Finance evidence is monthly equities/ETFs: DFL beats MSE on Sharpe (≈1.2–1.3 vs 0.7) but
  **polarises forecasts and concentrates portfolios**; the best results came from an
  **α-blend of DFL and MSE loss** ([arXiv 2409.09684](https://arxiv.org/html/2409.09684v4)).
- SPO models churn **84–95%/month** even at high risk aversion; fix is clipping + partial
  adjustment ([arXiv 2605.01176](https://arxiv.org/html/2605.01176)). Grade: medium-weak.
- **For us:** long-only with per-pair caps is a box/simplex projection with a closed form; a
  full optimisation layer is unnecessary. The transferable idea is the *blended loss* (brief 2).

## C. Reinforcement learning

| Work | Result | Grade |
|---|---|---|
| Zhang, Zohren & Roberts, *Deep RL for Trading*, JFDS 2020 ([arXiv](https://arxiv.org/pdf/1911.10107)) | 50 futures, 2011–19; vol-scaled return − bp·|Δpos| reward; DQN Sharpe 1.29 vs 0.06 long-only; positive to ~25 bp | **strong** (daily futures) |
| Kruthof & Müller, *Can deep RL beat 1/N?*, FRL 2025 ([link](https://ideas.repec.org/a/eee/finlet/v75y2025ics154461232500131x.html)) | SAC, 7 datasets, >300 yr OOS: does **not** beat 1/N frictionless; **negative net at 0.1% costs** | strong (negative) |
| Théate & Ernst, TDQN, ESWA 2021 ([arXiv](https://arxiv.org/pdf/2004.06627)) | Sharpe 0.40 vs 0.37 buy-and-hold at 0.1%; high seed sensitivity | medium |
| Gort et al., crypto PBO, 2022 ([arXiv](https://arxiv.org/pdf/2209.05559v4)) | 5-min, 10 coins, **8-week test**; "winning" PPO −35% vs −48% equal-weight; PBO method is the useful part | weak (result) / medium (method) |
| TradeMaster, NeurIPS 2023 ([paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/b8f6f7f2ba4137124ac976286eacb611-Paper-Datasets_and_Benchmarks.pdf)) | Generic TD3/DDPG ≥ trading-specific RL; near market average in crises; BTC minute set is ~1 month | medium |
| Bai et al. 2024 review ([arXiv](https://arxiv.org/abs/2411.12746)); Hoque et al. 2025 ([arXiv](https://arxiv.org/pdf/2512.10913)) | Shaped rewards help; **algorithm family doesn't matter**; most papers ignore costs; "implementation quality outweighs algorithmic complexity" | medium |
| Sun, Wang & An, ACM TIST 2023 ([PDF](https://personal.ntu.edu.sg/boan/papers/TIST23_RLFintech.pdf)) | Overfitting, distribution shift, baselines under-tuned | medium |
| Decision Transformer / CQL on DJIA ([arXiv](https://arxiv.org/html/2411.17900)) | DT ≈ imitates unvalidated expert policies; CQL better; no costs | weak |
| Henderson 2018; Agarwal 2021 ([rliable](https://github.com/google-research/rliable)) | Seed variance dominates deep-RL results; report IQM + bootstrap CIs | strong (method) |

**FreqAI's RL module** ([docs](https://www.freqtrade.io/en/stable/freqai-reinforcement-learning/),
[code](https://github.com/freqtrade/freqtrade/blob/develop/freqtrade/freqai/RL/BaseEnvironment.py)),
verified against the develop branch, Sept 2026:

- `Base3ActionRLEnv` (neutral/buy/sell, `can_short=False`) is the long-only choice; SB3 /
  sb3-contrib 2.8–2.9, gymnasium 1.2–1.3; Apple Silicon installs fine but **RL runs on CPU**
  (`BaseReinforcementLearningModel` bypasses the MPS device selection; SB3 PPO is CPU-first).
- **Default reward pays +25 per entry, −1 per idle candle, and `pnl × 100` on exit (scaled
  ×1.5 / ×0.5 by trade duration and boosted above the profit target)** → a round trip losing
  0.3% in fees costs well under 1 reward unit against +25 for entering. The docs call it a showcase and
  warn agents "find cheats". It must be replaced.
- **`add_state_info` (position, P&L, duration in the state) raises in backtesting**; a
  position-aware agent can only be validated in dry-run.
- `best_model.zip` is selected by mean shaped reward on the test split → selection leaks and
  optimises the wrong thing.
- One agent per pair on a sliding window: a 30-day 15-m window is 2,880 candles — tiny.
- Training env ignores `custom_exit` / `custom_stoploss` — i.e. the repo's ATR stop and grace
  logic, which `AGENT_GUIDE.md` identifies as the live levers.
- No public reproducible profitable FreqAI-RL strategy was found.

**Structural point** *(inference, supported by the surveys)*: a small spot trader is a
price-taker, so the market transition is exogenous to its actions. Any policy can be replayed
exactly on history; there is no exploration problem and no distribution shift from actions.
RL's machinery (value bootstrapping, off-policy correction) solves problems we do not have,
while adding variance we cannot afford at ~100 trades/yr. Direct gradient on the P&L (family
A) is the same objective with a lower-variance estimator. RL earns a slot only for
path-dependent decisions the differentiable simulator cannot express smoothly (discrete
stops, holding-time limits) — brief 4.

## D. Generative / diffusion / world models

| Work | Result | Grade |
|---|---|---|
| **CTBench** crypto time-series-generation benchmark, 2025 ([arXiv](https://arxiv.org/pdf/2508.02758)) | Hourly Binance, 452 coins 2020–24, train predictor on synthetic → test on real. **Diffusion-TS had the best fidelity and negative CAGR** ("accuracy–alpha gap"); winners flip by regime; 0.03% fees reorder rankings; TimeVAE trains in <1 min | medium |
| Coletta et al., *Conditional Generators for LOB Environments*, 2023 ([arXiv](https://arxiv.org/html/2306.12806)) | RL market-maker inside a CGAN simulator made "consistently large profits" by **exploiting the generator's order-imbalance artefact**; "may be unprofitable in real markets" | strong (negative) |
| LOB-Bench, ICML 2025 ([arXiv](https://arxiv.org/html/2502.09172v2)) | Best generators still discriminable (ROC 0.83); errors compound over long rollouts | strong (fidelity) |
| TRADES / DeepMarket diffusion LOB sim, 2025 ([arXiv](https://arxiv.org/html/2502.07071v3)) | Better than CGAN on train-synthetic/test-real; **~6 GPU-h per simulated hour** | medium |
| MarS (Microsoft), ICLR 2025 ([arXiv](https://arxiv.org/abs/2409.07486)) | 1–3B-param order-level model; only a rising in-sim training curve, no real-market validation | weak (for trading) |
| Paster et al., *You Can't Count on Luck*, NeurIPS 2022 ([arXiv](https://arxiv.org/abs/2205.15967)) | **Return-conditioned generative policies (Decision Transformer / Decision Diffuser) copy lucky trajectories in stochastic environments**; fix is conditioning on cluster-average return (ESPER) | strong |
| Diffuser / Decision Diffuser / Diffusion Policy | Strong on D4RL robotics; **no credible trading application found** | none (trading) |
| DARL diffusion-augmented PPO, 2025 ([arXiv](https://arxiv.org/pdf/2510.07099)) | Sharpe 1.91 vs 1.52; single bull window, no seeds | weak |
| Bayesian robust trading with adversarial generator + self-play, 2026 ([arXiv](https://arxiv.org/html/2601.17008)) | 9 daily ETFs 2021–24, Sharpe 1.07–1.86 vs 0.82–1.39; seeds/fees unreported. Best-designed generator+policy loop found | medium-weak |
| Tail-GAN, Mgmt Sci ([arXiv](https://arxiv.org/html/2203.01664v4)) | Judge a generator by whether it reproduces the **P&L tails of the strategies you care about** | strong (risk) |
| Shumailov et al., *Nature* 2024 ([link](https://www.nature.com/articles/s41586-024-07566-y)) | Recursively training generators on their own output collapses the tails first | strong |
| Time-series foundation models: Rahimikia et al. 2025 ([arXiv](https://arxiv.org/html/2511.18578v1)); Noguer i Alonso & Franklin 2026 ([arXiv](https://arxiv.org/abs/2606.27100)); Kronos AAAI 2026 ([arXiv](https://arxiv.org/html/2508.02739v1)) | Zero-shot Chronos/TimesFM **OOS R² −1.4% / −2.8%**; fine-tuning limited or harmful; gains over random walk "small and sparse"; Kronos captures volatility not conditional mean; indie 5-min BTC test ≈ Brownian | medium-strong (negative) |

**Takeaways.** No credible case of alpha transferring from a learned simulator or a synthetic
training set to real markets after costs. A generator cannot add information the 1.5M bars do
not contain, and our data is ~4 orders of magnitude below the only pretraining that showed
skill. The defensible use of the repo's DDPM/WGAN/CTAB-GAN+ generators is *stress testing*
a policy (brief 6). The "recursively trained" idea has no real instance beyond
fictitious self-play against an adversarial generator, and self-training on generated data is
actively harmful.

## E. Self-supervised, adaptive, evolutionary, LLM

**Self-supervised representations.** TS2Vec, TS-TCC, CoST, SimMTM, PatchTST-masked are
benchmarked on ETT/Electricity/UCR, not trading. Contrastive asset embeddings help *risk*
tasks (hedge selection, 19.1% vs 23.8% realised vol) and were not tested on returns
([Dolphin 2024](https://arxiv.org/html/2407.18645)). Nothing credible shows an SSL embedding +
small head beating gradient-boosted trees on hand-built features for direction after costs.
Grade: weak; expect null. Cheap to test (brief 7).

**Regime discovery.** Statistical jump models (Shu, Yu & Mulvey 2024,
[arXiv](https://arxiv.org/html/2402.05272v2)): S&P Sharpe 0.48 → 0.68, max DD −55% → −27%
with online inference, 1-day delay, 10 bp; turnover 44% vs 141% for an HMM. The authors call
it "interpretative rather than predictive" — the benefit is drawdown. Changepoint features
raised Deep Momentum Network Sharpe ~⅓ ([Wood 2021](https://arxiv.org/abs/2105.13727v3)).
**Leakage trap:** smoothed vs filtered HMM probabilities on the same strategy: Sharpe 0.78 vs
1.74 ([demo](https://github.com/dmitridefreitas-dev/regime-detection)); `hmmlearn.predict`
is Viterbi over the whole sequence ([issue](https://github.com/hmmlearn/hmmlearn/issues/540)).
The repo's own Phase C (`regime/VERDICT.md`) found no vol/Hurst/autocorr/breadth gate beat
the fixed choice — consistent with the literature that regime gates buy drawdown, not return.
Grade: medium (risk side). Brief 8 is a negative control.

**Online / continual / meta-learning.** FSNet/OneNet update on labels not yet observable;
PROCEED (KDD'25, [arXiv](https://arxiv.org/html/2412.08435v5)) shows errors roughly double
under realistic label delay. A leak-free re-evaluation shrinks online-adaptation gains by
**11–40pp** ([arXiv 2609.01126](https://arxiv.org/html/2609.01126)). DoubleAdapt (KDD'23)
recommends combining with monthly full retraining. FreqAI `continual_learning`: "Experimental",
"high probability of overfitting / getting stuck". Grade: unclear-to-negative. Excluded as a
standalone project; scheduled rolling retraining is the default in every brief.

**Evolutionary / dynamic structure.** Allen & Karjalainen JFE 1999: GA rules earn no
consistent excess return after costs (strong, negative). NEAT on S&P 2025: 18.8% vs 28.0%
buy-and-hold, no costs. **"The Red Queen's Trap" 2025** ([arXiv](https://arxiv.org/html/2512.15732v1)):
500 evolved LSTM/Transformer agents on 1-m/15-m Binance top-20 perps — **>300% validation APY,
>70% live capital decay**; 51.2% directional accuracy vs ~55% breakeven at 0.08% round-trip.
"Model complexity cannot compensate for information deficiency." MadEvolve 2026 (LLM-guided
evolution on BTC minute bars): validation-to-test gap widens with search-space size. ES/CMA-ES
on backtest P&L: valid gradient-free policy search, **no rigorous OOS evidence**, and every
candidate is a trial for PBO purposes. Grade: weak-to-negative. Brief 5 confines it to the
non-differentiable pieces with strict trial accounting.

**Meta-labelling.** Singh & Joubert / Hudson & Thames, E-mini 2011–19: OOS Sharpe 0.18 → 0.40
(mean-reversion), 0.26 → 0.32 (trend) ([PDF](https://hudsonthames.org/wp-content/uploads/2022/04/Does-Meta-Labeling-Add-to-Signal-Efficacy.pdf));
validation F1 gains (0.29 → 0.56) were far larger than OOS (0.26 → 0.30). "If the algorithm is
bad, meta-labeling would likely only reduce the downside." Grade: medium-weak but consistent,
and directly compatible with the repo's trade archive (brief 3).

**LLM agents.** FINSABER (KDD 2026, [arXiv](https://arxiv.org/abs/2505.07078)): 20 years,
>100 symbols — FinMem/FinAgent advantages "deteriorate significantly". *Profit Mirage*
([arXiv](https://arxiv.org/abs/2510.07920v1)): returns evaporate after the model's knowledge
cutoff (memorised history). Grade: strong (negative). Excluded.

## F. Evaluation methodology (applies to all)

- Deflated Sharpe Ratio (Bailey & López de Prado 2014) and Probability of Backtest Overfitting
  via CSCV (Bailey et al. 2015, [PDF](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf),
  [`pypbo`](https://github.com/esvhd/pypbo)).
- CPCV beats walk-forward on both PBO and DSR in a controlled study (Arian, Norouzi & Seco,
  KBS 2024, [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4778909)) — the repo's
  pooled anchored walk-forward is kept as the arbiter because it matches production retraining,
  but CPCV is a reasonable second opinion for the label-free methods, which overfit more easily.
- Seed reporting: IQM + stratified bootstrap ([Agarwal 2021](https://arxiv.org/abs/2108.13264)).
- Data reality *(inference)*: 800 days × 96 bars × 20 correlated pairs ≈ 2.2 years of one
  market; SE of an annualised Sharpe estimate ≈ 1/√2.2 ≈ 0.67. No single window can tell a
  Sharpe-1 strategy from luck. This is the same conclusion `regime/WALKFORWARD_POWER.md`
  reached empirically.

## G. What transfers to a 15-min, long-only, liquidity-capped, ~10 bp/side spot bot

1. Costs, spread and fill caps in the objective; evaluate at full cost; sweep training-cost
   scale γ (DeePM).
2. Sigmoid exposure (or softmax with a cash slot) through a **no-trade band**; explicit
   turnover penalty; minimum holding periods. Avoid straight-through / Gumbel discrete actions —
   no trading evidence they beat soft positions.
3. Utility loss (mean − λ/2·var of net log returns) or pooled Sharpe with a SoftMin
   worst-window term; drawdown/CVaR only as regulariser.
4. Small models (single LSTM / small Mamba / small transformer), truncated BPTT with burn-in,
   5–10 seed ensembles.
5. Semi-supervised warm start from the existing classifier or auxiliary volatility heads;
   α-blended loss.
6. Generators for stress-testing only; foundation models as optional features with a null
   prior; no online fine-tuning; no return-conditioned planners; no LLM agents.
7. Every trial counted; DSR/PBO reported; pooled walk-forward as the arbiter.
