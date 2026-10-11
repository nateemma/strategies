# Contracts: Regime Gate on the Reversion Family (026)

## C1 — `utils/RegimeState.py` (strategy-agnostic, pure)

- `breadth(frames: dict[str, DataFrame], ema=768, min_pairs=6) -> Series`
- `volreg(frames, win=96, trail=2880, min_pairs=6) -> Series`
- `persist(frames, q=16, win=2880, min_pairs=6) -> Series`
- `btctrend(btc: DataFrame, ema=768) -> Series`
- `INSTRUMENTS = {id: callable}`, `compute(id, frames, btc) -> Series`

Frames have `date` and `close` columns. Outputs are indexed by `date` (UTC, 15m), causal, and NaN
where undefined. There are no pair names in the logic beyond dict iteration.

## C2 — `research/regime_gate/feasibility.py` (US1; no backtests)

```
python -m research.regime_gate.feasibility --primary st_fam_combo --cross st_gbb \
    --out specs/026-reversion-regime-gate/feasibility.md
```

- Loads the 10 whitelist pairs and BTC 15m (`research.clfstudy.data.load_pair`).
- Computes the instruments (C1).
- Pools the arms (`evaluation.pool`), takes their reversion-family trades (`family_of(enter_tag)`;
  all of `st_gbb`), and looks up each instrument at the signal bar (`open_date − 15m`).
- Chooses the GateRule on f00–f07 (R2) and scores FeasibilityRecords (R3).
- Writes markdown and `feasibility.json`.
- Exit 0 if ≥1 PASS, 3 if none pass (STOP), 2 on missing inputs.
- The rule choice must not read late-fold trades. A unit test plants an instrument that separates
  only the late half perfectly and asserts it is not chosen as PASS through the early choice.

## C3 — SimpleTrigger gate (US2)

- `REVERSION_REGIME_GATE` (data model). `informative_pairs()` adds BTC/USDT 15m when needed.
- In `get_entry_conditions` / `_tags`:
  1. state = the instrument series (C1) over `dp.current_whitelist()` frames (and BTC), cached per
     last date, aligned to this pair's dates;
  2. `off = rule(state)`, NaN → False;
  3. drop rows where the winning family is reversion and `off`;
  4. overlap rows in `off` are tagged momentum.
- Sink: `regime_gate_refused` and `regime_gate_off_bars` (025 C3 format).
- `None`: byte-identical to 025.
- **Warm-up (H1):** a gate requires `startup_candle_count >= 4096`, or class creation is refused.
  The control `SimpleTrigger_RegOn4096` (gate None, startup 4096) establishes whether the longer
  startup is itself neutral.

## C4 — Fingerprint

`REVERSION_REGIME_GATE` is added to the optional family members when non-None. Arm D,
`SimpleTrigger_Gbb` and `SimpleTrigger_FamCombined` keep their shas (test).

## C5 — Stage (reuse 025 `evaluation/stage.py`)

`--stage regime --previous st_fam_combo --candidates <gated arms>`, with a SECOND judgement against
`st_bgo_mx` (FR-008).
- Adoption requires late IMPROVED against the better constant. The arm compared against is the one
  with the higher late-half sum (here `st_bgo_mx`).
- **Required changes to `stage.py`:**
  - a `--bar` option naming the arm to judge against, defaulting to `--previous`;
  - **(H2)** for `--stage regime`, the degeneracy rule checks only the momentum family (both
    halves) and reversion's early half; other stages are unchanged.
