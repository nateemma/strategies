# Schemas

## ExperimentConfig (YAML entry in a phase file)
```yaml
- arch: LSTM_FAM
  hp: {hidden: 64, depth: 1, bidirectional: false}
  input_set: ind24
  seq_len: 16
  split_set: v1
  train: {max_epochs: 40, patience: 10, batch_size: 256, lr: 0.0001, weight_decay: 0.0, stride: 1}
  phase: 4
  seeds: [0, 1]
  tags: [family-sweep, lstm, size]
```
`config_id = sha1(json.dumps({arch, hp, input_set, seq_len, split_set, train}, sort_keys=True, separators=(",", ":")))[:12]`

## SplitSet file (`splits/v1.json`)
```json
{"split_set": "v1", "embargo_bars": 96, "val_fraction": 0.10, "min_history_days": 365,
 "timeframe": "15m", "pairs": ["ZEC/USDT", "..."],
 "windows": [{"window_id": "A", "train_start": "2022-01-01T00:00:00Z", "val_start": "...",
              "train_end": "2023-06-30T00:00:00Z", "test_start": "2023-07-01T00:00:00Z",
              "test_end": "2023-12-31T23:45:00Z", "distant": true}, "..."]}
```

## RunResult (one line of `results/results.jsonl`)
Exactly the fields in `data-model.md` §RunResult. Numeric NaN is written as JSON `null`.
Lists/matrices are JSON arrays. `hp` and `train` are embedded objects.

## GateDecision (`gates/phase_N.json`)
```json
{"phase": 4, "rule": {"margin_floor": 0.02, "margin_used": 0.031, "std_bound": 0.03, "majority_of": 3},
 "ridge_reference": {"A": {"mcc": 0.21, "macro_f1": 0.40}, "B": {...}, "C": {...}, "pooled_seed_std_mcc": 0.0155},
 "per_config": {"3f9a...": {"A": {"mcc_ok": true, "f1_ok": true, "std_ok": true, "status_ok": true}, "B": {...}, "C": {...}, "viable": true}},
 "viable_set": ["3f9a..."], "decided_at": "...", "decided_by": "gate.py + author review"}
```
