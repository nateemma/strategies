# Contract: Trial log (`research/labelfree/trials.jsonl`)

Append-only JSONL. One JSON object per line, schema per data-model §1.

## Protocol

1. **Before** a configuration runs, append its record with `"result": null` (FR-019, SC-015).
2. After it finishes, append a second record with the same `trial_id` carrying the result, or
   rewrite in place — either is acceptable provided the pre-run record's existence remains
   verifiable.
3. Abandoned or crashed runs get `result: {status: "abandoned", reason: ...}`. **Records are never
   deleted** — deletion is what the deflation statistics exist to defeat.

## Audit (final task of the feature)

The audit MUST assert:
- every configuration evaluated on any fold has a record;
- every record was written before its result existed;
- `trial_id` is unique;
- the trial count used for DSR and PBO equals the record count (FR-024);
- the pinned trend-rule baseline contributes no records (FR-021a).

A failed audit invalidates the overfitting statistics and therefore the verdict.

## Example (pre-run record)

```json
{"trial_id":"t0007-utility-band-g1.0-s3-f07","written_at":"2026-09-10T14:02:11Z",
 "objective":"utility","turnover_control":"band","gamma":1.0,"lambda_risk":2.0,
 "architecture":{"kind":"lstm","hidden":64,"seq_len":16},"band_width":0.10,"seed":3,
 "fold_id":"f07","spread_estimator":{"kind":"abdi_ranaldo","floor_bp":1.0},
 "liquidity_cap":{"fraction":0.10,"soft_min_tau":0.01},"leaky":false,"result":null}
```
