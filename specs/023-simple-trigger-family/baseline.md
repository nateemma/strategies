# 023 baselines

## T002 — reproducibility on HEAD before any 023 edit (2026-10-10, HEAD 8dd041e)

Run from the outer repo root with the freqtrade env loaded (`set -a; source ~/.freqtrade_env;
set +a` — without it config validation fails on `api_server.jwt_secret_key` and `run_arm`
reports "no new result file"; see memory `reference_jwt_secret_blocks_all_runs`). Command shape
copied from `evaluation/run_folds.py:evaluate_fold`.

```bash
PYTHONPATH=user_data/strategies .venv/bin/python -m regime.run_arm --strategy NNNC_ARO_f00 \
  --timerange 20230101-20230401 --label F00 --tag base023_aro \
  --config user_data/strategies/config/config_wf.json
PYTHONPATH=user_data/strategies .venv/bin/python -m regime.run_arm --strategy GbbOpen \
  --timerange 20230101-20230401 --label F00 --tag base023_gbbo \
  --config user_data/strategies/config/config_wf.json
```

| arm | recorded | rerun | trades | verdict |
|---|---|---|---|---|
| arm D f00 (`wf_aro_f00_F00`) | `5c53c833e423eae4` | `5c53c833e423eae4` | 52 / 52 | MATCH |
| `wf_gbbo` f00 (`wf_gbbo_f00_F00`) | `2d59b026e4cb9797` | `2d59b026e4cb9797` | 544 / 544 | MATCH |

## T017 — shared edits are neutral (2026-10-10)

Edits under test: `Framework/BaseStrategy.py` (entry-condition sink), `regime/run_arm.py`
(framework/sink/exit-reason record fields, `--no-sink`), `evaluation/{run_folds,pool,foldplan,
compare_pooled}.py`. Same commands as T002, tags `gate023_*_{sink,nosink}`.

| arm | sink (run_arm default) | `--no-sink` | baseline | verdict |
|---|---|---|---|---|
| arm D f00 | `5c53c833e423eae4` (52) | `5c53c833e423eae4` (52) | `5c53c833e423eae4` | MATCH |
| `wf_gbbo` f00 | `2d59b026e4cb9797` (544) | `2d59b026e4cb9797` (544) | `2d59b026e4cb9797` | MATCH |

`framework_sha`: arm D `303456c1fa322b4c`, GbbOpen `8284f40a913a6c54` (they differ — research R1).

First liveness readings (f00, guards-open operating point): every quality guard and `rvol` pass
100% of bars (open by design); the binding conditions are the signal and `volume > 0` (~64% of
bars pass — the binanceus gap-fill). arm D signal 304 bars → 206 after volume; `gbb` trigger
10,511 → 7,769. Bar totals differ by (200−64)×9 = 1,224, the startup_candle_count difference.

`compare_pooled.compare` refactored into `compare_data` + formatter: output byte-identical on 4
comparisons under a fixed `PYTHONHASHSEED`. **Pre-existing defect found, NOT fixed:** the paired
bootstrap is not reproducible across processes — `bootstrap_delta` builds its diff list by
iterating Python sets, whose order follows hash randomisation. On `rung_aro_tp0.022` vs arm D
the CI bounds move ±0.02pp/yr and p ±0.01 between processes. Decision for the user (the file
forbids retuning; a fix — sorting the keys — shifts every recorded value slightly).

## T032 — live dry-run isolation (SC-008)

`python evaluation/dump_strategy.py --strategy MomentumRegimeBasket15mFast` (every public class
attribute across the MRO + merged params JSON; the strategy has no JSON) run in a `git worktree`
at `9e82a45` (pre-implementation) and in the current tree: **identical** (63 lines). The dry-run
chain is `MomentumRegimeBasket15mFast -> MomentumRegimeBasket15m -> IStrategy`; it never imports
BaseStrategy, SimpleTrigger or the harness.
