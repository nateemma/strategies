# Contract: ExecutionRecorderMixin (`utils/ExecutionRecorder.py`)

A strategy-agnostic mixin. Its home is utils/, not Framework/.

```python
class ExecutionRecorderMixin:
    RECORD_EXECUTION: bool = False         # default OFF
    RECORD_DIR: str | None = None          # root; run_id subdirectory appended
    RECORD_RUN_ID: str | None = None
    RECORD_DEPTH_LEVELS: int = 20
    MODEL_KEY_IGNORED = ("RECORD_EXECUTION", "RECORD_DIR", "RECORD_RUN_ID", "RECORD_DEPTH_LEVELS")
```

## Guarantees

1. **Pass-through when inactive.** Inactive means `RECORD_EXECUTION` is false, or `dp.runmode` is
   not in {live, dry_run}. Then every overridden method returns exactly what `super()` returns,
   with no I/O, so backtest output is byte-identical (test: the trades hash of a backtest with the
   mixin equals the one without it).
2. **Never changes a decision.** Each override calls `super()` first, records, and returns the
   parent's value. Recording errors are caught and logged, and never propagate into trading.
3. **Same model key.** A class with the mixin resolves to the same feature-030 key as the class
   without it (test).
4. **Callbacks and what they record:**

   | callback | records |
   |---|---|
   | `custom_stake_amount` | remembers `proposed_stake` per pair for the next entry record |
   | `confirm_trade_entry` | an OrderRecord plus an order DepthSnapshot (if super returns False, the record has `rejected = true`) |
   | `order_filled` | a FillRecord |
   | `confirm_trade_exit` | an ExitRecord plus an exit DepthSnapshot |
   | `bot_loop_start` | a scheduled DepthSnapshot per whitelist pair, at most one per pair per candle |
   | `populate_entry_trend` | a SignalRecord for the last closed candle of the pair |

5. **Idempotent.** On start the mixin loads today's keys, and duplicate keys are skipped. Files
   are append-only and fsynced per record.
6. **Bounded:** each depth request has a 3 s timeout. On failure it records `depth_missing` and
   never retries inside the trading loop.
7. **Where data goes:** all paths are under `RECORD_DIR`, which must resolve outside the
   strategies repository (refused otherwise).
