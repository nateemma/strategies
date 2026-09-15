# Data Model: labelfree-ssl-features

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

**The bar is not a constant.** Measured before this design (research.md R1):

```
full 24-feature production set      H=6  rho_oos 0.1594      H=48  rho_oos 0.0914
the 6 panel indicator columns       H=6  0.055-0.092         H=48  0.056-0.069
```

The quoted ρ≈0.15 reproduces at **horizon 6 on the full feature set** and nowhere else. The spec
pre-declared a constant 0.15 at **horizon 48**, where the hand-built features themselves reach
0.0914 — a bar the baseline cannot clear, so an embedding failing it would fail for a reason
unrelated to the hypothesis. Every screen therefore carries its own bar, measured on the same
data, pipeline and horizon.

```
passed = (rho_embedding > rho_handbuilt)  and  interval excludes rho_handbuilt
         and (rho_control << rho_embedding)
```

**The PRIMARY combination is `engineered` × horizon 48**, declared before any screen runs. The
other three are secondary and exploratory: a pass there is a hypothesis for a follow-up, never an
acceptance. Four screens against one threshold is four chances to clear a bar meant to be cleared
once — the failure brief 05 measured at PBO 0.7128.

---

## 5. `PairAgnostic` — the deployability gate, pre-registered

For each embedding dimension `d`:

```
ratio(d) = std_over_pairs( mean_within_pair(d) )  /  mean_over_pairs( std_within_pair(d) )
reject if  max over d of ratio(d)  >  0.5
```

Between-pair variation in *location* exceeding half the within-pair *scale* means the dimension
does not mean the same thing on every pair, which `AGENT_GUIDE.md:545` prohibits.

**A ratio rather than an absolute**, because embedding dimensions have arbitrary units.
**Pre-registered at 0.5 before any encoder is trained**, for the same reason feature 013's
liveness thresholds were: a threshold chosen after seeing the distribution is not a threshold.

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
