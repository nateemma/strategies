# NNMT operating point — headroom, and what it forecloses

**Status**: settled 2026-09-06 (feature 005). Supersedes nothing; complements
`GUARD_ABLATION.md` and `CANDIDATE_STATUS.md`, which established the same shape
on NNNC.

**One-line answer: NNMT's ENTRY model does not select its trades and a perfect
one would do worse — but its multi-task SELL head is live, and the single
positive result on this family is on the exit side.**

> **UPDATE, later on 2026-09-06.** The entry-side conclusion below stands
> unchanged. After fixing `exit_rvol_threshold`'s resolution, a 2-D exit sweep
> found `enable_exit_signal=True` + `exit_rvol_threshold=1.0` worth **+0.78pp on
> W1**, positive on all three windows across two GENUINE seed replicates
> (spread 0.08pp). See `specs/005-nnmt-operating-point/exit_side.md`. This does
> not weaken any entry-side foreclosure — the oracle measured `predict_buy` and
> says nothing about the exit path (R9).

## The measurement

Control = NNMT's pre-inheritance configuration (label 0.007, take-profit 0.032),
pinned in `NNMT/NNMT_MLX_Base.json`. Windows: **W1 `20250901-20260831` is the
sole acceptance basis**; W2 `20240601-20250531` and W3 `20230101-20231231` are
in-sample. Training `20220101-20250831`. Resolution throughout is a paired 2sd
bootstrap over each arm's symmetric difference with its control — **not** a flat
1pp floor, which would have overstated oracle-arm resolution ~4.7×.

| W1 arm | trades | return |
|---|---|---|
| no model at all (guards alone) | 179 | +10.689% |
| **real model** | 176 | **+10.708%** |
| perfect model *(LOOKAHEAD CEILING, not achievable)* | 155 | +6.150% |

- **Model contribution: +0.019pp across 7 differing trades** (2sd = 1.05pp).
- **Headroom: −4.558pp on W1**, negative on all three windows (−2.93 W2,
  −1.09 W3). Its magnitude sits inside its own 4.66pp band, so the defensible
  claim is *"certainly not materially positive, directionally negative 3 of 3"*
  — not *"significantly worse"*.

## Why: the guards are the strategy, and they interact

Leave-one-out over all six quality guards, W1:

- **`entry_guard_threshold` is INERT** — byte-identical trades in all three
  windows. NNMT's swept 0.3 is meaningless.
- `entry_rvol_threshold` is nearly inert at its setting (176 → 174 trades). Note
  it is declared `decimals=1`, so the swept **0.25 actually executes as 0.2**.
- `adx` dominates (176 → 542 trades), `atr_pct` second (176 → 392).
- **No individual guard is resolved on return**, but the joint effect is
  −34.150pp against a −8.123pp sum of individuals — **4.20× super-additive**.

The chain is a strongly interacting conjunction. **A leave-one-out cannot rank
these guards and must not be used to justify opening any one of them.**

## The label: a 46pp ceiling move worth −0.16pp of realised return

The inherited change 0.007 → 0.02 was measured at the ceiling (free) and then
with four trained models (`MLX_SEED` 42/7; `train_seed` is inert for multi-task).

| | label 0.007 | label 0.02 |
|---|---|---|
| W1 guards-off oracle *(ceiling)* | **−28.113%**, 2126 tr — gate **FAIL** | **+17.791%**, 1561 tr — gate **PASS** |
| W1 realised, threshold re-swept per config | +10.386% (2 seeds) | +10.229% (2 seeds) |

**Label effect −0.157pp, against a between-seed spread of 0.813pp.** Seed 42
gives *byte-identical trades* at thresholds 0.5 and 0.7.

Across the 20 W1 sweep cells, **8 are byte-identical to the production model's
trade set**, including a model trained on the other label. On W1 the twelve
cells in the 0.4–0.6 threshold plateau collapse into **two** distinct trade sets.
(W1-specific: on W2 eleven of twelve are distinct, on W3 nine of twelve — the
models differ in mechanism there, but return indistinguishably.)

Take-profit 0.032 → 0.022: **−0.258pp on W1 against 2sd 0.53pp**, sign flipping
positive on both in-sample windows. Null.

## The chosen configuration: unchanged

**No parameter is changed by this feature.** Both inherited changes are adopted
as they stand — the label because it is already configured, passes the standing
oracle gate, and costs nothing measurable; the take-profit because it is not
resolvably worse than the value it replaced. Neither was a defect. The
inconsistency that motivated the feature is real but **BENIGN**.

All eleven pre-existing NNMT models were trained at 0.007 and are **STALE** with
respect to the configured 0.02. They are deliberately not retrained: the label
demonstrably makes no difference to realised return at this gate.

## What this FORECLOSES (the point of this document)

A future proposal for model-side or augmentation work on NNMT can be **rejected
without new measurement** if it relies on any of the following:

1. **"A better model will improve entries."** No. A *perfect* entry signal is
   worth −4.56pp on W1, and bypassing the model entirely costs 0.019pp. There is
   nothing above the current model to capture.
2. **"A better label / target will help."** No. A label change that moved the
   ceiling by 46pp moved realised return by −0.16pp against 0.81pp of seed noise,
   and produced byte-identical trades at two of three thresholds tested.
3. **"A better architecture / more capacity will help."** No — and this now
   holds for a second, structurally different model class. NNNC's nine-variant
   null was on `MLXClassifierNary`; NNMT is `MLXClassifierMultiTask` with six
   heads. The gate dominates on both.
4. **"A path-aware / stop-aware label will help."** No — built and tested
   2026-09-07. `LabelMethod.gbb_path` drops buy labels the stop would pre-empt
   and sell labels the take-profit would pre-empt. It PASSED the oracle gate
   (+23.991% guards-off against the incumbent's +17.791%) and improved the
   guards-on W1 ceiling by +1.971pp — and realised **−0.091pp** on W1 across two
   seeds. See `regime/PATH_LABEL_GATE.md`.
5. **"GAN augmentation will help."** No. Augmentation improves what the model
   learns; the model's opinion does not reach the trades. This closes the
   augmentation follow-on for NNMT specifically, at this gate.
6. **"Tune the confidence threshold."** No. Thresholds 0.4–0.6 are
   indistinguishable and largely byte-identical on W1; 0.7–0.8 only removes
   trades.
7. **COCONUT stays closed.** It was re-openable only if NNMT's headroom proved
   materially LARGER than NNNC's 0.68pp. It is smaller — negative. Condition
   explicitly not met.

**What is NOT foreclosed — and one of the two has now paid.**

**(a) The EXIT side.** Every foreclosure above concerns the ENTRY model. The
sell head is a different object and it is demonstrably live: two seeds produce
identical W1 entries but DIFFERENT exits in every window. `exit_signal=True` with
`exit_rvol_threshold=1.0` is +0.78pp on W1. Note the two halves are worthless
apart — the signal alone at the shipped gate is −0.12pp, and the gate alone is
dead code — which is why both were previously filed as nulls.

**(b) The entry GATE, not the entry model.** Every result in this repository that moved P&L was a gate
or mechanism change. On NNMT that means the guard chain — but note guards-off at
the current label costs −34.15pp, and the label-0.02 ceiling of +17.791% at an
open gate is a *ceiling*, not a forecast. Any such attempt must first demonstrate
that a trained model discriminates once the guards stop binding; on NNNC that
required opening the guards AND raising the threshold together, and NNNC's
values are explicitly not transferable here.

## Scope of the claim

This is a statement about **NNMT at its current gate on these three windows**,
not about the family in all configurations. It rests on W1 as the sole
acceptance window; W2 and W3 corroborate direction only, and **W3 is near-flat
on this family** (126 control trades, +1.238%, PF 1.118) so nothing here leans
on it. Every ceiling figure is a lookahead bound and none is promotable.
