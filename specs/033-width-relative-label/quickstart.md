# Quickstart: validating feature 033

All commands run from `user_data/strategies` unless noted. Load the environment first:
`set -a; source ~/.freqtrade_env; set +a`.

## 1. Byte-identical when off (SC-001)

```
../../.venv/bin/python -m pytest Framework/tests utils/tests SimpleTrigger/tests research/exit_coupling/tests \
  research/held_pair/tests research/candidate_training/tests research/width_relative/tests evaluation -q -p no:cacheprovider
```

Expect:
- the suite passes (check pytest's own exit code);
- `test_pooled_label.py` shows the spec-absent build equal to the 032 fixture;
- the model keys equal `model_keys_pre033.json`.

## 2. The label on one fold (US1, SC-002/SC-003)

Train ARO f00 alone (`research/width_relative/train.py --model ARO --fold f00`), then read
`width_relative.json` beside the weights. Expect:
- 10 buckets, each with a positive share in [0.40, 0.60];
- |Spearman(label, bb_width)| ≤ 0.05;
- SELL = 0 in `row_filter.json`;
- the existing `NNNC_ARO_f00` weights unchanged (sha).

## 3. Sizer instrument on f00 (contract C2 checks)

Expect:
- the flat-stake twin's `trades_hash` equals `w_aro` f00's;
- the thresholds load to 4 decimals;
- tag-to-agree mismatches 0.

## 4. The study (after the pre-registration commit)

1. Training chain, one fold per process: ARO, then GBR.
2. Threshold picks, one fold per process.
3. Plans, then `evaluation.run_folds`, one chain.
4. `research/nnnc_framework/signals --models WR_ARO,WR_GBR`.
5. Instrument (S0), the separation screen (S1), the judge (S2/S3).

Each stage reads only what the earlier stages allow. Expected output: `verdicts.json` and
`report.md` with PASS (PROVISIONAL) / FAIL / INCONCLUSIVE per model.
