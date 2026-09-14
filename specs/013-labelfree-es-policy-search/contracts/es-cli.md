# Contract: `research/labelfree/es/` command surface

Every command is sequential (FR-021), deterministically seeded, and appends to the sidecar
before it scores (FR-013). All are run as `python -m research.labelfree.es.<module>` from
`user_data/strategies`.

---

## `space.py` — inspect the search space

```
python -m research.labelfree.es.space --show
```

Prints the 10 dimensions, their bounds, the incumbent value and the incumbent's source file.
`--check` round-trips `decode(encode(x)) == x` over 10,000 random points and exits non-zero on
any mismatch.

---

## `exits.py` — the exit surrogate (library, not a CLI)

```python
def simulate(exposure, prices, atr_pct, rvol, params, costs) -> TradeSeries
```

Replays a fold's exposure path bar by bar, opening a position when exposure turns on and closing
it on the **first** of: the ATR-adaptive stop (volume-confirmation tightened, grace-gated),
`cexit_take_profit`, `cexit_max_days`, or the replayed `model_exit` / `unclog_12h`.

Returns per-trade entry/exit bar, price, exit reason and net P&L after the brief-1 cost model.

`unclog_12h`, `unclog_1d` and `max_hold` are **recomputed**, not replayed — all three are purely
per-trade (`BaseStrategy.py:1211-1218`). This is what makes `hold_window_hours` (dimension 10,
minimum holding time) searchable at all.

**Explicitly not modelled** (research.md R3): `model_exit` is **replayed from the archive**,
because it depends on the model's per-bar prediction, which is not a searched parameter. It is
11.9% of trades, so a parameter change that would have altered when the model exits is invisible
to the surrogate. This is stated in the report.

---

## `calibrate.py` — THE BLOCKING GATE

```
python -m research.labelfree.es.calibrate --n 40 --folds f05..f14 --out calibration.json
```

Scores 40 space-spanning candidates with both the surrogate and a real freqtrade backtest
(~175 s each; ≈2 h total). Writes `spearman_rho`, `win_disagreements` and a pass/fail.

**Exit code is non-zero when the gate fails**, and every downstream command refuses to run
without a passing `calibration.json`. An uncalibrated surrogate does not risk a wrong answer —
it means the entire search budget optimises a fiction.

---

## `liveness.py` — Principle IV over the search space

```
python -m research.labelfree.es.liveness --folds f05..f14 --out liveness.json
```

Sweeps each dimension alone across its bounds with the other 9 at the incumbent, and reports the
objective's span per dimension against the **pre-registered** thresholds of data-model.md §6
(`inert` < 0.10 pp/yr, `weak` < 3.35 pp/yr, else `live`). Runs **before** the search.

---

## `run.py` — the per-fold driver

```
python -m research.labelfree.es.run --fold f07 --method cmaes  --budget 1600 --seed 7
python -m research.labelfree.es.run --fold f07 --method random --budget 1600 --seed 7
```

Searches on the fold's **training range only** (FR-002), then evaluates the single best candidate
on the fold's **test range exactly once** (FR-010) and stops. `--budget` is a count of
evaluations, not wall-clock (research.md R5), and is asserted identical across the two methods.

Refuses to start if `calibration.json` is missing or failing.

---

## `report.py` — PBO, deflation and the result document

```
python -m research.labelfree.es.report --out research/label-free-study/briefs/results/05-2026-09-13.md
```

Reads the **full sidecar** — every candidate, every generation, both methods — and emits:

- the **protocol §5 verdict**: pooled annualised return vs the production control against the
  3.35 pp/yr floor with a bootstrap interval (SC-001), per-fold positivity (SC-002), W1/W2/W3
  single-window persistence (SC-002), the four mandatory baselines against the +7.86 pp/yr bar
  (SC-003), and any configuration whose in-sample advantage exceeds its held-out advantage by
  more than 3.35 pp/yr (SC-008)
- **PBO as the headline number**, with `n_evaluations` beside it (FR-016)
- the **unregularised** score beside the regularised one, so the reader can see what the
  incumbent-distance and turnover penalties did (data-model.md §1a)
- DSR deflated by the **total** evaluation count (FR-015)
- every return carrying trade count, trades/yr and mean holding time (FR-017)
- the delta against the incumbent measured against the **4.93 pp** sweep-noise band; inside it,
  the wording is "not resolved", never "improvement" (FR-018)
- any parameter sitting on a bound, reported as a space drawn too narrowly (FR-019)
- the liveness report's inert dimensions

Written **whether the verdict is accept or kill** (FR-020).

---

## `audit` — the leakage proof

```
python -m research.labelfree.es.history --audit
```

From the sidecar alone, without trusting the search code:

1. exactly one `range == "test"` evaluation per fold (SC-011)
2. every test evaluation's `written_at` is later than every train evaluation for that fold
3. every evaluation has a `written_at` preceding its score (FR-013)
4. the shared `trials.jsonl` summary row's line count and sha256 match the sidecar (SC-018)

Exits non-zero on any violation.
