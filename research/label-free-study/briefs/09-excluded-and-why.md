# Brief 09 — Approaches considered and excluded

Recorded so they are not re-proposed without new evidence. Each line names the strongest
reason; the review (`01-literature-review.md`) has the citations.

| Approach | Why excluded | Would reconsider if… |
|---|---|---|
| **Diffusion / Decision-Transformer planners** (Diffuser, Decision Diffuser, return-conditioned DT) | Provably copy lucky trajectories in stochastic environments (Paster et al. 2022); no credible trading application; sampling cost ~100× per step | someone shows ESPER-style cluster-conditioning working on real market data after costs |
| **Training a policy inside a learned market simulator** (Dreamer-style world model, CGAN/diffusion LOB sims, MarS) | Agents exploit simulator artefacts (Coletta 2023); no alpha-transfer record; TRADES needs ~6 GPU-h per simulated hour; our data is ~4 orders of magnitude below the scale that showed skill | a published sim-to-real trading result with costs and multi-year OOS |
| **Recursive self-training on generated data** | Collapses tails first (Shumailov, *Nature* 2024) — the tails are the risk | never for generators; see brief 6 for the legitimate use |
| **Fine-tuning time-series foundation models** (Chronos, TimesFM, Kronos, TimeGPT) | Zero-shot OOS R² negative; fine-tuning limited or harmful in two studies; captures volatility, not conditional mean | zero-shot *features* are tested in brief 7; fine-tuning only if that screen is positive |
| **FreqAI `continual_learning` / online fine-tuning** | Docs: experimental, overfit-prone; leak-free re-evaluations shrink online-adaptation gains by 11–40pp; label-delay leakage is endemic (PROCEED) | a leak-free study shows a gain ≥ the floor on crypto intraday |
| **FreqAI RL module as shipped** | Churn-rewarding default reward; state-aware agent not backtestable; test-split model selection; per-pair sliding-window agents | the module is only used as a reference; brief 4 builds its own env |
| **NEAT / genetic programming of rules / full-network neuroevolution** | Negative after costs since Allen & Karjalainen 1999; "Red Queen's Trap" (15-m crypto, +300% val → −70% live); PBO explodes with population size | brief 5 keeps the narrow, trial-accounted version |
| **LLM trading agents** (FinAgent, FinMem, TradingGPT, "AQuA"-style research loops) | Advantage disappears on 20-yr/100-symbol tests (FINSABER); returns evaporate post knowledge-cutoff (*Profit Mirage*); implausible reported ICs | a post-cutoff, cost-inclusive, multi-year live record |
| **Offline RL (CQL / IQL)** | Solves action-coverage distribution shift, which a price-taker does not have; no cost-inclusive intraday evidence | brief 4's PPO fails for a reason offline RL addresses |
| **Full decision-focused optimisation layers** (cvxpylayers / SPO+) | Long-only with caps is a closed-form projection; DFL polarises forecasts and churns 84–95%/month; no Apple-GPU solver | constraints become genuinely non-trivial (e.g. cross-pair risk budgets) |
| **Pseudo-labelling / co-training / label propagation** | Fix label *scarcity*; every bar already has a forward return — the problem is signal-to-noise, not missing labels | never, in this formulation |
| **Autoencoder anomaly score as an entry signal** | Evaluated on reconstruction quality only; no OOS trading evidence; the repo's `Anomaly/` family already exists and did not change the picture | as a risk-off feature inside brief 8 |
| **More hand-crafted labels** (breakout, trend-scanning, multi-horizon votes) | `regime/` already showed learnability and tradeability move oppositely and that the ceiling is a model/objective problem; the trend-scanning "win" in the literature had acknowledged calculation errors | never as the primary route; labels re-enter only as the prior in brief 2 |
