# Data Model: labelfree-ssl-features

> ### ⚠️ CORRECTED 2026-09-15 — the numbers below are LEAK ARTEFACTS
>
> Every ρ figure in this document was measured with a forward-return target starting at
> `open[t]`, which **included bar *t*'s own move** — information the inputs reveal directly.
> The causally aligned target is `open[t+1] → open[t+1+H]`, and under it the hand-built bar is:
>
> ```
> H        3        6       12       24       48       96
> median  0.0352   0.0461   0.0396   0.0328   0.0238   0.0401
> ```
>
> **Two claims stated here as findings do not survive.** "ρ ≈ 0.15 is a horizon-3-to-5 number"
> — under causal alignment nothing reaches 0.15. And "predictability decays with horizon" — it
> is roughly **flat** at 0.024–0.046 from H=3 to H=96; the decay was the leak's own signature,
> since bar *t*'s move is proportionally the largest part of a short-horizon target.
>
> The text is left standing rather than rewritten, because a corrected document should show what
> it corrected. Current numbers: `research/labelfree/ssl/baseline.json`. Full history of the
> three generations: [`CEILING_CORRECTIONS.md`](../../research/labelfree/CEILING_CORRECTIONS.md).

Phase 1. Entities, fields, and the invariants that make the gates checkable.

---

## 1. `Input` — the two things the encoder can consume

| input | content | why it is here |
|---|---|---|
| `engineered` | the 24-column `FeatureNormalizer.include_list` matrix from `features.build_features` | cannot add information the indicators lack, so a null means SSL finds no exploitable structure the classifier misses |
| `ohlcv` | raw open/high/low/close/volume windows | the strongest form of the hypothesis — and the input a learned 1D-CNN already failed on at ρ 0.04–0.06 |

Both are reported separately (FR-007a). They fail for different reasons and conflating them would
lose which.

**`ohlcv` needs normalising before it is an input at all.** `AGENT_GUIDE.md:546` prohibits raw
price and raw volume as NN inputs because their ranges differ by pair. The window is therefore
converted to log returns and a volume ratio before the encoder sees it — which is the same rule
the production features already obey, applied to the encoder's input rather than its output.

---

## 2. `Encoder` — TS2Vec in MLX

| component | choice | why |
|---|---|---|
| backbone | dilated **causal** CNN, 10 levels | causal by construction, which is what makes FR-002 satisfiable rather than merely checkable |
| output width | 320 | the TS2Vec default; **not searched** — searching it across four screens is the multiple-testing failure R7 guards against |
| loss | hierarchical contrastive: temporal + instance-wise, at successive pooling levels | the paper's construction |
| views | two random crops per instance | |
| seed | `MLX_SEED` | a framework-level `train_seed` was measured inert for multi-task models in this repo |

**Fitted per fold, on the training range only, without labels** (FR-001).

**Every updated array goes into the per-step `mx.eval`** (FR-005). Omitting one lets the graph
grow until Metal's ~500K limit trips — a failure this repository has already recorded.

---

## 3. `Split` — three ways, in time order, within a fold

```
fold training range   →  A  encoder      (no labels)
                      →  B  ridge        (supervised; labels from B only)
fold test range       →  C  score        <- the only number reported
```

**Why three and not two.** There are two fittable objects and the obvious design leaks through
the second. The encoder is unsupervised, so it is tempting to fit the ridge on the same range —
but the ridge *is* supervised, and one fitted and scored on the same bars beats any bar. That is
how a null becomes a false positive.

**Purge and embargo apply at every boundary**, not only A|B: a label at the end of B reaches
`horizon` bars into C.

---

## 4. `Screen` — and the bar is measured, not quoted

For each of the four (input × horizon) combinations, three numbers computed **identically**:

| quantity | what it is |
|---|---|
| `rho_embedding` | nested-OOS Spearman of the ridge projection of the embedding |
| `rho_handbuilt` | **the bar** — the same ridge over the 24-column production feature set |
| `rho_control` | the same ridge over a same-width random Gaussian block |

**The bar is not a constant — it is a function of horizon.** Measured across 5 folds × 6
horizons, 10 pairs each (research.md R1):

```
H       3        6       12       24       48       96
median  0.1717   0.1127   0.0944   0.0720   0.0826   0.0583
```

**ρ ≈ 0.15 is a horizon-3-to-5 number**, not a general ceiling; predictability decays roughly
monotonically with horizon. The spec pre-declared a constant 0.15 at **horizon 48**, where the
hand-built features reach **0.083** — a bar the baseline cannot clear, so an embedding failing it
would fail for a reason unrelated to the hypothesis. Every screen therefore carries its own bar,
measured on the same data, pipeline and horizon.

```
passed = (rho_embedding > rho_handbuilt)  and  interval excludes rho_handbuilt
         and (rho_control << rho_embedding)
```

**The PRIMARY is `engineered` passing at BOTH horizon 48 and horizon 6** — amended after the bar
was measured (research.md R7), because at H=48 the hand-built bar is 0.0499 with a negative
minimum and "beat that" is nearly "beat zero". Declared before any encoder was trained. The
other three are secondary and exploratory: a pass there is a hypothesis for a follow-up, never an
acceptance. Four screens against one threshold is four chances to clear a bar meant to be cleared
once — the failure brief 05 measured at PBO 0.7128.

---

## 5. `PairAgnostic` — the deployability gate, pre-registered

For each embedding dimension `d`:

```
ratio(d) = std_over_pairs( mean_within_pair(d) )  /  mean_over_pairs( std_within_pair(d) )
reject if  max over d of ratio(d)  >  0.30
```

Between-pair variation in *location* exceeding half the within-pair *scale* means the dimension
does not mean the same thing on every pair, which `AGENT_GUIDE.md:545` prohibits.

**A ratio rather than an absolute**, because embedding dimensions have arbitrary units.

**0.30 is anchored on production, measured across four folds** — production's worst single
dimension reaches 0.2297 (f10), so 0.30 gives ~1.3× headroom. An earlier draft said 0.5, which is
over 4× production's typical maximum: a gate nothing plausible could fail, and therefore worse
than no gate, since a decorative PASS reads as corroboration.

**Pre-registered before any encoder is trained.** Anchoring on *production's* distribution is
legitimate — production is not the thing under test — while a threshold chosen after seeing the
*embedding's* distribution would not be a threshold at all.

**A rejection is a finding** (FR-012b) — an embedding that cannot be deployed here, however well
it screens — not something normalised away.

---

## 6. `Causality` — deletion, plus a planted leak

| check | construction | passes when |
|---|---|---|
| deletion | recompute a bar's embedding with every later bar removed | bit-equal to the full-series embedding |
| planted leak | inject the forward return as an extra input channel | the screen's ρ **jumps** |

The architecture is causal by construction, which is an argument rather than a measurement. And
**a causality test that cannot detect a leak it was handed is not evidence** — feature 011 used
exactly this pair.

---

## 7. `Verdict`

```
rho_embedding, rho_handbuilt, rho_control   per (input, horizon)
primary                                     "engineered@48"
pair_agnostic_max_ratio, pair_agnostic_pass
causality_deletion_pass, planted_leak_detected
status   "KILL" | "SCREEN PASSED — BACKTEST EARNED" | "NOT RESOLVED"
```

Derived, never typed:

- primary `rho_embedding` ≤ its own `rho_handbuilt` → **KILL**, and the measured ρ is the finding
- `rho_control` ≈ `rho_embedding` → **KILL**, recorded as capacity rather than information
- pair-agnostic rejection → **KILL** for deployability, reported as a finding about the method
- causality deletion fails, or the planted leak is undetected → **the screen is void**, not a result
- otherwise → the backtest is *earned*, and protocol §5 decides
