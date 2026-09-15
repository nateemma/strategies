# Contract: `research/labelfree/ssl/` command surface

Sequential, from `user_data/strategies`, under `freqtrade/.venv/bin/python`.

---

## `ts2vec.py` — fit the encoder (library + CLI)

```
python -m research.labelfree.ssl.ts2vec --fold f07 --input engineered --seed 7
```

Dilated **causal** CNN + hierarchical contrastive loss, MLX. Fitted on the fold's **training
range only, without labels**. Writes `embeddings/<fold>_<input>_s<seed>.npz`.

Width 320 and the remaining hyperparameters are **TS2Vec defaults and are not searched** —
searching them across four screens is the multiple-testing failure the design guards against.

Every updated array is in the per-step `mx.eval`; omitting one trips Metal's ~500K limit.

---

## `causality.py` — the deletion test and the planted leak

```
python -m research.labelfree.ssl.causality --fold f07 --input engineered
```

Deletion: a bar's embedding recomputed with every later bar removed must be **bit-equal**.
Planted leak: injecting the forward return as an extra channel must make the screen's ρ **jump**.

**Exit non-zero if either fails.** A causality test that cannot detect a leak it was handed is not
evidence, and a screen run on an uncausal embedding is void rather than negative.

---

## `pairagnostic.py` — the deployability gate

```
python -m research.labelfree.ssl.pairagnostic --embeddings <npz> --out pairagnostic.json
```

Rejects if any dimension's between-pair location spread exceeds **0.5×** its within-pair scale
(pre-registered, data-model.md §5). A rejection is reported as a **finding**: an embedding that
cannot be deployed in this repo however well it screens.

---

## `screen.py` — THE BLOCKING GATE

```
python -m research.labelfree.ssl.screen --folds f05..f14 --out screen.json
```

For each of the four (input × horizon) combinations, three numbers computed identically on the
same three-way split: `rho_embedding`, `rho_handbuilt`, `rho_control`.

**The bar is measured, not quoted.** `rho_handbuilt` is the same ridge over the 24-column
production feature set at the same horizon on the same data. Measured beforehand: the full set
reaches 0.1594 at H=6 and 0.0914 at H=48, so a constant 0.15 would be a bar the baseline itself
cannot clear at the primary horizon.

**Primary is `engineered` × horizon 48**, fixed before any screen runs. A secondary that passes is
a hypothesis for a follow-up, never an acceptance.

Exit non-zero when the primary does not beat its own bar — **and no backtest may run** (FR-010).

---

## `report.py` — the §6 result document

```
python -m research.labelfree.ssl.report --out research/label-free-study/briefs/results/07-<date>.md
```

Reports all four combinations with their own bars, the control, the pair-agnostic ratio and the
causality results. Written **whatever the outcome** (FR-016), and a KILL at the screen is the
expected path.

States that the screen is a **learnability** measure and therefore
**necessary-but-not-sufficient**: this repo has recorded a +0.155 MCC producing identical trades.
Passing it earns a backtest, never an acceptance.
