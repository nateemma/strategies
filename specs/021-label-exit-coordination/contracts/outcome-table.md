# Contract: the per-bar outcome table

**FR-006 through FR-009b.**

## Commands

```
python -m research.outcomes.table --build [--pairs ...] [--timerange ...]
python -m research.outcomes.table --status      # what is cached, and under which config hash
python -m research.outcomes.table --verify      # rebuild one pair, assert identical (SC-003)
```

## Storage

`research/outcomes/.cache/<pair>__<config_hash>.feather`, with
`research/outcomes/.cache/<config_hash>.json` recording the full `ExitConfig`.

## Cache-key discipline

The hash covers every parameter `simulate_forward` reads, **including `fee_per_side`** — a
positive outcome is net of fees, so a fee change changes every outcome. Requesting a table
whose hash does not match the requested config is an **error**, never a silent reuse
(FR-007). The sidecar exists so staleness can be diagnosed by reading it.

## Leakage guard (FR-009) — two layers, both tested

1. **Reserved prefix.** Every outcome column is named `__outcome_*`. Model input is built
   from `FeatureNormalizer.include_list`, so a prefixed column cannot be selected by accident.
2. **A raising assertion at the boundary.** The label builder asserts no column it passes
   onward carries the prefix, and a test feeds an outcome-bearing frame into the label path
   and asserts it raises.

A convention alone would not be "structurally prevented". The prefix makes accidental
selection impossible; the assertion makes deliberate misuse loud.

## Acceptance

- SC-003: two builds from the same inputs are identical
- SC-004: changing any exit parameter forces a rebuild
- FR-009b: economic value reported both over all triggered bars and fillable-only
- FR-020 / SC-009: `__outcome_gapfill_share` recorded per record, since 22–55% of 15m bars on
  six of eleven production pairs are gap-fill and outcomes computed across them describe the
  absence of trading
