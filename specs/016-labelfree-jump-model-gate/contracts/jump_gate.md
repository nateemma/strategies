# Interface Contract: `research/labelfree/jump/`

Internal research module. The contract is the Python surface other modules and the tests bind to.
Entry point named by the brief: `research/labelfree/jump_gate.py` (thin, delegates here).

---

## `features.py`

```python
def build(pair: str, fold: Fold) -> pd.DataFrame
```
Causal `rvol_realised`, `rvol_downside`, `sign_persistence` on closed bars.
**Guarantee**: row `t` is a function of bars `≤ t` only.

---

## `model.py`

```python
@dataclass(frozen=True)
class JumpFit:
    centroids: np.ndarray      # (K, d)
    weights:   np.ndarray      # (d,)
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    lam: float
    k: int
    state_order: tuple[int, ...]

def fit(x: np.ndarray, k: int, lam: float, *, seed: int = 0,
        max_iter: int = 100) -> JumpFit

def fit_ensemble(x: np.ndarray, k: int, lam: float,
                 seeds: Sequence[int] = (0, 1, 2, 3, 4)) -> list[JumpFit]

def seed_agreement(fits: list[JumpFit], x: np.ndarray) -> float
```
Three-block coordinate descent (centroids, states, sparse weights).
**Guarantees**: `fit` is deterministic for a given `(x, k, lam, seed)` and **stochastic across
seeds** through its k-means initialisation; `state_order` sorts states by fitted
realised-volatility centroid ascending.

**Contract**: callers MUST use `fit_ensemble` (≥5 seeds, FR-022) and report the interquartile
mean — never a single `fit`, never the best seed. `seed_agreement` must return **≥ 0.95** or the
feature stops (FR-023).

```python
def calibrate_lam(x: np.ndarray, k: int, target_switch_rate: float) -> float
```
Called **once**, on fold 0's training window. Its result is frozen for the whole run.

---

## `online.py` — the file the whole feature turns on

```python
def filtered_states(x: np.ndarray, fit: JumpFit) -> np.ndarray
```
Forward recursion only: `s_t = argmin_k V_t(k)`.
**Guarantee**: `filtered_states(x[:m], fit) == filtered_states(x, fit)[:m]` for every `m`.
This equality is the causality contract and is asserted by test, not assumed.

```python
def smoothed_states(x: np.ndarray, fit: JumpFit) -> np.ndarray
```
Forward pass to `T`, then **back-track** from `argmin_k V_T(k)`. The leak reference.
**Guarantee**: the prefix equality above **must fail** for this function. A `smoothed_states`
that satisfies it is not running the traceback and is not a leak reference.

---

## `refit.py`

```python
def schedule(fold: Fold, period_days: int = 90) -> list[pd.Timestamp]
def run(panel: pd.DataFrame, fold: Fold, lam: float, k: int) -> pd.DataFrame
```
`run` returns a `StateSequence` frame (`pair, timestamp, state, mode, refit_id`), refitting on the
expanding window and **continuing through the fold's test range** (clarification Q2=A).
**Guarantees**: each fit sees only bars strictly before its refit point; states are
volatility-remapped at every refit so labels never swap.

---

## `crosscheck.py`

```python
def audit_package_inference() -> dict     # {'filtered': bool, 'evidence': str}
def agreement(ours: np.ndarray, theirs: np.ndarray) -> float
```
`agreement` **below 0.95 stops the feature** (FR-016). The in-repo model is primary and carries
the verdict; the package corroborates it and never overrides it (FR-014).
`audit_package_inference` must return `filtered=True` **before** `agreement` is meaningful.
Both run under `freqtrade/.venv/bin/python`, asserted via `sys.executable`.

---

## `redundancy.py` — runs before any performance number (Principle IX)

```python
def overlap(decisions: pd.DataFrame) -> dict
# {'n_blocked', 'n_rvol_blocked', 'n_both', 'novel_block_rate', 'verdict'}
```
`verdict` = `'redundant_with_rvol'` when `novel_block_rate < 0.20`, else `'distinct'`
(FR-024, SC-019).

---

## `gate.py`

```python
def decide(states: pd.DataFrame, high_vol_state: int) -> pd.DataFrame   # GateDecision rows
def activity(decisions: pd.DataFrame, fold_id: str) -> dict             # GateActivity
```
`activity['verdict']` ∈ `{'active', 'inert', 'total'}`.
**Contract**: callers MUST check for `inert`/`total` before computing any performance statistic.

---

## `arms.py`

```python
def generate(fold: dict, states_path: Path, arm: str) -> tuple[Path, Path]
def assert_reuses_weights(gated_cls: str, ungated_cls: str) -> None
```
`generate` writes the gated fold class and its params file. The class overrides
`get_model_path()` to return the **ungated** fold's directory.
`assert_reuses_weights` raises unless `saved_data/<ungated>/*.safetensors` exists and no
`saved_data/<gated>/` directory was created. Must be called before evaluation.

---

## `drawdown.py`

```python
def equity_pp(trades: list[dict]) -> np.ndarray        # additive, close-ordered
def max_drawdown(curve: np.ndarray) -> float
def time_underwater(curve: np.ndarray, span_bars: int) -> float
def bootstrap_ci(trades: list[dict], stat: Callable, block_len_trades: int,
                 n: int = 10_000, seed: int = 0) -> tuple[float, float]
```
`bootstrap_ci` reuses `research.labelfree.stress.bootstrap.block_index`.
**Contract**: `block_len_trades` is in **trades**; passing a bar-panel constant is a unit error.

---

## `report.py`

```python
def write(results: dict, out: Path) -> Path
def register_trials(records: list[dict]) -> None
```
`register_trials` wraps `research/labelfree/trials.py::pre_register` / `complete` so this
feature's trials enter the log the deflated Sharpe ratio is computed over
(`00-evaluation-protocol.md` §5 gate 4).
Writes `research/label-free-study/briefs/results/08-<date>.md`. Runs **whatever the outcome**
(FR-021), including when the feature stopped early at the leak test or the cross-check.
