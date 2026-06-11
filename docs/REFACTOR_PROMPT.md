# Claude Code prompt — class hierarchy refactor & consolidation

> Paste everything below the line into Claude Code, run from the repo root with the
> official `code-review` plugin installed and `ast-grep` (`sg`) on PATH.

---

You are refactoring the freqtrade strategy code under `user_data/strategies/`. First read
`user_data/strategies/AGENT_GUIDE.md` and follow it. **Hard constraint: only touch files under
`user_data/strategies/` — the rest of the freqtrade tree is re-cloned and changes outside this path are lost.**

## Context

This is a deep, organically-grown inheritance tree that I want cleaned up. Known shape:

- Base chain: `BaseNNStrategy → NNNCStrategy → NNNC_CGP → NNNC_CGP_MLX → {Attention, CNN, GRU, LSTM, Mamba, TSMamba, Transformer, Wavenet, …}`
- Parallel branches off `NNNCStrategy`: `NNNC_MLX`, `NNNC_WGAN`, `NNNC_PCA`, `NNNC_DDPM_*`
- Sibling families in other dirs: `Anomaly/`, `GANs/`, `Predictors/`, `TSPredict/`, `NNPredict/`, `NNMT/`, `Sklearn/`, `SimpleStrategies/`, plus shared `Framework/` and `utils/`.
- ~396 Python files. The hierarchy and naming drifted over a long period.

## Goals, in priority order

1. **Fix the class hierarchy.** Find where inheritance is wrong, redundant, or upside-down:
   duplicated method overrides that are identical to the parent, deep chains that could be
   flattened, copy-paste between sibling subclasses that belongs in a shared base/mixin, and
   layering that violates the dependency direction (a base importing from a subclass dir, etc.).
2. **Consolidation.** Identify near-duplicate classes/modules that differ only in a config value,
   a model name, or a few lines — candidates to merge into one parameterized class, or to replace
   with a mixin/composition. Quantify: how many files, how many lines could collapse.
3. **Efficiency.** Flag obvious perf issues in the hot paths (`populate_indicators`,
   `populate_entry/exit_trend`, feature building, any per-candle loops that could be vectorized,
   repeated recomputation that could be cached, redundant DataFrame copies).

## How to work

- **Map before you touch anything.** Use `ast-grep` for structural facts, not grep/regex. Examples:
  - class + base list: `sg run -p 'class $NAME($$$BASES): $$$BODY' --lang python`
  - method overrides: `sg run -p 'def populate_indicators($$$): $$$' --lang python`
  Build a picture of the full inheritance graph (who extends whom, across dirs) and an index of
  duplicated method bodies.
- Produce a written **analysis + plan first** as `user_data/strategies/docs/REFACTOR_PLAN.md`:
  the current hierarchy (as a tree), a ranked list of consolidation opportunities with file/line
  counts and risk level, the efficiency findings, and a proposed target structure. **Stop there and
  let me approve before editing.**
- After I approve, work in **small, independently reviewable batches** (one family or one
  consolidation at a time). For each batch: make the change, then run the `/code-review` plugin on
  the diff and address high-confidence findings before moving on.
- **Preserve behavior.** These are trading strategies — don't change signal logic or
  hyperparameters while refactoring. If a "duplicate" actually differs in a meaningful value, flag
  it, don't silently merge. Keep class names that are referenced by configs/backtests stable, or
  give me an explicit rename list.
- If there are tests or a backtest harness, run/identify them and tell me how to verify a batch
  didn't change results.

## Deliverables

1. `docs/REFACTOR_PLAN.md` — analysis, ranked opportunities, target structure (await my approval).
2. Batched refactor PRs/commits, each `/code-review`-clean.
3. A short `docs/REFACTOR_CHANGELOG.md` listing every class moved/merged/renamed and why.

Start by reading the guide and building the inheritance map. Show me the map and the plan; do not edit code yet.
