# Interface Contract: `research/signalctx/`

Internal research module. The contract is the Python surface tests and callers bind to.

---

## `signals.py`

```python
def reconstruct(fold_id: str, train: bool = False) -> pd.DataFrame   # Signal rows
def fidelity(fold_id: str) -> dict    # {'n_trades','matched','rate','misses'}
```
`reconstruct` collapses contiguous `enter_long` runs to one Signal (R2).
**Contract**: callers MUST check `fidelity(fold)['rate'] >= 0.95` before using a fold.
Measured f08: 70/71 = 0.986, the miss being the test range's first bar.

---

## `context.py`

```python
def build(signals: pd.DataFrame, panels: dict) -> pd.DataFrame       # SignalContext
HYPOTHESES: tuple[str, ...]   # exactly 4, fixed (R5)
ALPHA: float                  # 0.05 / len(HYPOTHESES)
```
**Guarantee**: `build(signals[:m], ...)` equals `build(signals, ...)[:m]` for every `m`.
`prev_signal_won` is null unless the previous trade **closed** before this signal.

---

## `paths.py`

```python
def unrealised(trade: dict, panel: dict, mark: str = "close") -> pd.Series
def reconcile(trade: dict, path: pd.Series) -> float   # |path[-1] - recorded| in profit_ratio
RECONCILE_TOL: float = 0.002        # profit_ratio units; ~2x a round-trip fee
```
**Contract**: `reconcile` must return **≤ `RECONCILE_TOL`**, else the reconstruction is wrong and
nothing downstream is evidence.

---

## `confound.py` — US1, the gate

```python
def strata(paths: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame
def cmh(table) -> tuple[float, float]                  # statistic, p  (primary)
def logit(df) -> dict                                  # secondary
def verdict(res: dict) -> str                          # 'survives' | 'proxy_for_drawdown'

STAKE_PP_PER_RATIO: float = 9.0     # 900 stake on a 10,000 wallet (brief 013)

def effect_sizes(df) -> dict
# {'effect_winrate_pp': ..., 'effect_return_pp_yr': ...}
```
**Contract**: `verdict` gates on **`effect_return_pp_yr`** only. Passing `effect_winrate_pp` to
the 7.633 pp floor is a units error — the two are different currencies, and the motivating
14.8-point win-rate effect would clear that floor while carrying no return information. The
implementation MUST raise rather than silently compare the wrong one.

---

## `sweep.py` — US3

```python
def test_hypotheses(ctx: pd.DataFrame, exploratory: bool) -> pd.DataFrame
```
Returns one row per hypothesis with statistic, p, **both effect sizes**, and pass/fail at
`ALPHA`. `exploratory` is **read from `out/confound.json`**, not passed in memory, so the label
cannot be lost between processes (FR-023).

---

## `masked.py` — US4, fenced

```python
def observe(fold_id: str) -> pd.DataFrame     # MaskedOutcome rows
def assert_not_pnl(df: pd.DataFrame) -> None  # raises if used on a P&L path
```
`observe` runs the stacked backtest. Every row carries the dry/live label.
**Contract**: `assert_not_pnl` is called by `report.py` before writing any return figure.

---

## `arms.py` — US2/US5

```python
def generate(fold: dict, rule: str) -> tuple[str, Path, Path]
def assert_reuses_weights(cls: str, base: str) -> None
def assert_entries_identical(arm_archive, control_archive) -> None
def liveness(archive, base_archive) -> dict   # {'n_triggered', 'identical_to_control'}
def overlap_with_stop(archive, within_bars: int = 8) -> float   # >= 0.70 is inert
TIGHTEN_FACTOR: float = 0.5   # frozen; an outright exit is a sensitivity only
```
**Contract**: `liveness` reports `n_triggered` AND whether the arm is byte-identical to its
control. `n_triggered == 0` with a non-identical arm means the probe is broken, not that the
rule is inert (R9).

---

## `report.py`

```python
def write(results: dict, out: Path) -> Path
def register_trials(records: list[dict]) -> None
```
Runs whatever the outcome, including an early stop at US1.
