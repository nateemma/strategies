# Quickstart / Validation Guide: labelfree-jump-model-gate

Every command runs from `user_data/strategies`, **sequentially**, under
`/Users/philprice95/projects/freqtrade/.venv/bin/python`. The pyenv shim is a different
interpreter without pandas or scipy.

```bash
PY=/Users/philprice95/projects/freqtrade/.venv/bin/python
$PY -c "import sys; print(sys.executable)"     # must print the .venv path
```

## 0. Dependency

```bash
$PY -m pip install jumpmodels          # 0.1.1, pure wheel, no new transitive deps
$PY -c "import jumpmodels; print(jumpmodels.__version__)"
```

## 1. Unit tests (run before anything expensive)

```bash
$PY -m pytest research/labelfree/tests/test_jump_*.py -q
```
Expected: all pass. The decisive ones:

| test | what it proves |
|---|---|
| `test_jump_causality::test_filtered_prefix_stable` | truncating history leaves filtered states bit-identical |
| `test_jump_causality::test_smoothed_prefix_CHANGES` | the leak reference really does leak |
| `test_jump_model::test_state_order_stable` | labels do not swap across refits |
| `test_jump_drawdown::test_additive_not_compounded` | concurrent trades are not compounded |

## 2. Seed ensemble — the fit is stochastic even though the backtest is not

```bash
$PY -m research.labelfree.jump_gate seeds --seeds 0,1,2,3,4
```
Expected: cross-seed state agreement **≥ 0.95**, and the per-fold block rate reported as an
interquartile mean with a stratified-bootstrap interval. Below 0.95 the regime structure is a
k-means initialisation artifact and the feature stops (FR-022, FR-023, SC-017).

Every later step consumes this ensemble, never a single fit.

## 3. Cross-check and the package audit (cheap, and can stop the feature)

```bash
$PY -m research.labelfree.jump_gate crosscheck
```
Expected: `package_mode_audited=True (filtered)` and an agreement rate against the in-repo model.
**Agreement below 0.95 stops here** (FR-016) — it is not averaged over.

## 4. Redundancy against the incumbent's rvol guard (Principle IX, before performance)

```bash
$PY -m research.labelfree.jump_gate redundancy --fold all
```
Reports `novel_block_rate`. Below **0.20** the verdict is `redundant_with_rvol`, meaning the gate is inert because rvol
got there first — a defect report, **not** evidence that regime gating does not help.

## 5. Gate activity (Principle IV liveness)

```bash
$PY -m research.labelfree.jump_gate activity --fold all
```
Expected: `active` in ≥10 of 15 folds with per-fold block counts.
`inert` → KILL, recorded as "the gate never fired".
`total` → refused by `admissibility.py`; abstention is not risk reduction.

## 6. Weight reuse check (before any backtest)

```bash
$PY -m research.labelfree.jump_gate check-weights --fold all
```
Expected: every gated class resolves to the **ungated** fold's `*.safetensors`, and no new
`saved_data/` directory exists. A failure here means a class would train inside its own
backtest — silently, with returncode 0.

## 7. The leak test — run before the honest number

```bash
$PY -m research.labelfree.jump_gate leaktest --fold all
```
Expected: the smoothed arm beats the online one by **≥ 7.633 pp** (the pinned noise floor).
That gap is the published leak size. Below it, stop — the online path is itself leaking
(FR-009, SC-014).

## 8. Gated walk-forward (30 backtests, no retraining, sequential)

```bash
$PY -m research.labelfree.jump_gate evaluate --arm gated_online   --fold all
$PY -m research.labelfree.jump_gate evaluate --arm gated_smoothed --fold all
```
The ungated baseline is **read**, never re-run: `evaluation/pooled_nnnc.json`, 15/15 folds,
378 trades, floor 3.350 pp/yr.

## 9. Drawdown with intervals

```bash
$PY -m research.labelfree.jump_gate drawdown --fold all
```
Expected: per-fold max drawdown and time-underwater with block-bootstrap CIs, beside return.
SC-011 needs drawdown to fall with a CI excluding zero in **≥10 of 15 folds** while return is
unchanged within the floor.

## 10. Report (runs whatever the outcome)

```bash
$PY -m research.labelfree.jump_gate report
```
Writes `research/label-free-study/briefs/results/08-<date>.md`.

---

## End-to-end expectation

The pre-registered expectation is a **KILL** consistent with `regime/VERDICT.md`. A negative
result that satisfies the constitution's workflow gates is a successful outcome and is recorded
with the same weight as a positive one.

The one result that would be a **defect rather than a finding** is an inert gate — and steps 3
and 5 exist to catch it before it is mistaken for a clean null.
