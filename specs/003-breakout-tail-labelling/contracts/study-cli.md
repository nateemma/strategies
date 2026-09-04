# Contract — Study CLI and Record Schemas

The study exposes no library API. Its interfaces are script invocations and the
JSON arm records they emit.

---

## 1. Screening — `regime/screen_candidates.py` (NEW)

```
PYTHONPATH=user_data/strategies .venv/bin/python \
  user_data/strategies/regime/screen_candidates.py \
  [--pairs XRP_USDT,SOL_USDT,LINK_USDT] [--out <path>]
```

**Guarantees**
- Loads no model and trains nothing (FR-003).
- Emits one ScreeningRecord per candidate with all six measures populated.
- Sets `excluded_reason` on any candidate below the density floor (FR-002).
- Sets `requires_hold_change` where time-to-peak exceeds the forced-exit window.

**Output** — markdown table plus one record per candidate:

```json
{
  "candidate_id": "L1_h48_t02",
  "density_pct": 5.81, "median_gain_pct": 4.29, "clears_stop_pct": 40.6,
  "reward_risk": 0.86, "time_to_peak_h": 8.2, "stopped_first_pct": 5.5,
  "n_pairs": 3, "requires_hold_change": false, "excluded_reason": ""
}
```

Exit code 0 always — exclusion is a result, not an error.

---

## 2. Arm execution — `regime/run_arm.py` (EXISTS, unchanged)

```
PYTHONPATH=user_data/strategies .venv/bin/python \
  user_data/strategies/regime/run_arm.py \
  --strategy <CLASS> --window W1|W2|W3 --tag <tag> \
  [--set KEY=VALUE] [--set-sell KEY=VALUE] [--roi V] [--threshold T]
```

**Guarantees relied on**
- `--window` resolves to a pinned literal timerange, never today-relative.
- Emits `n_trades`, `trades_hash`, `exit_config_hash`, `wallet_profit_pct`,
  `max_drawdown_pct`, `profit_factor`, `win_rate` to
  `regime/arms/<tag>_<window>.json`.
- A strategy with no params file falls back to CLASS defaults — which is why
  FR-016 requires one per arm.

**Known non-guarantee**: `--stoploss` is INERT for NN strategies. The operative
stop is the ATR band, whose knobs are class attributes unreachable via `--set`.
A stop sweep requires subclasses that also override `get_model_path()` to reuse
the parent's weights.

---

## 3. Training

```
cd <repo root> && PATH=.venv/bin:$PATH \
  zsh user_data/strategies/scripts/test_strat.sh -f -t 20220101-20250831 NNNC <CLASS>
```

**Contract**
- `-f` DELETES the existing model before retraining. Back up anything that
  matters first.
- `-t` sets the training window; it MUST end before W1 begins.
- Runs 8–45 min depending on label density. Execute SEQUENTIALLY.
- Kill by process pattern, not by task stop — the wrapper survives and respawns.

---

## 4. Comparison report

Every arm comparison MUST render, in this order:

1. `exit_config_hash` arm vs control — mismatch invalidates the comparison.
2. `trades_hash` per window — identical on W1 ⇒ verdict `inconclusive_absorbed`.
3. W1 return and trade count, marked OUT-OF-SAMPLE.
4. W2/W3 returns and counts, marked IN-SAMPLE and excluded from the verdict.
5. The verdict, with sub-power deltas stated as indistinguishable.

Classifier metrics MAY be recorded as context and MUST NOT appear as the basis of
a verdict (FR-014).
