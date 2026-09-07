# T018 — the two inherited changes: both verdicts (US2)

Each change measured ALONE against the pre-inheritance control, on two seeds,
judged on realised return on W1 only. Resolution is T008's paired 2sd bootstrap
over each arm's symmetric difference, not a flat floor. **No classifier metric
is an acceptance input here** (FR-019).

Four models were trained for this: `NNMT_Ctl_s42`, `NNMT_Ctl_s7` (label 0.007)
and `NNMT_Lbl02_s42`, `NNMT_Lbl02_s7` (label 0.02, profit head pinned), via
`MLX_SEED` because `train_seed` is inert for multi-task (R8). 66/59/46/37
minutes each.

---

## Change 1 — the label gain threshold, 0.007 → 0.02

### Verdict: **ADOPT 0.02** (already the configured value; no config change needed). The change is UNRESOLVABLE on realised return, and free.

The confidence threshold was re-swept per configuration (FR-021: a different
label shifts the confidence distribution, so a fixed threshold is confounded).
Each configuration's threshold was chosen on the **in-sample** windows only —
max mean of W2 and W3 — so W1 stays clean as the acceptance basis.

| label | seed 42 | seed 7 | mean | between-seed spread |
|---|---|---|---|---|
| 0.007 (pre-inheritance) | +10.148% @0.8 | +10.623% @0.7 | +10.386% | 0.475pp |
| 0.02 (inherited) | +10.635% @0.7 | +9.822% @0.8 | +10.229% | 0.813pp |

**Label effect on W1: −0.157pp. Between-seed spread: 0.813pp — 5× larger.**

Paired resolution, per seed:

| comparison | effect | differing trades | 2sd | verdict |
|---|---|---|---|---|
| seed 42 @0.5 | — | **0** | — | **INCONCLUSIVE — identical trades** |
| seed 42 @0.7 | — | **0** | — | **INCONCLUSIVE — identical trades** |
| seed 7 @0.8 | −0.236pp | 5 | 0.99pp | within noise |

### The liveness evidence is stronger than the returns

Across the 20 W1 sweep cells (4 configs × 5 thresholds), **8 are byte-identical
to the Phase-1 control** — the production `NNMT_MLX` model's own trade set.
Those 8 include `NNMT_Lbl02_s42` at thresholds 0.5 and 0.6: **a model trained on
a completely different target produced exactly the same trades as production.**

On W1 the twelve (config × threshold) cells in the 0.4–0.6 plateau collapse into
just **two** distinct trade sets. A fixed-label seed change (`Ctl_s7` vs
`Ctl_s42` at 0.5) is likewise byte-identical.

**This is W1-specific and must not be overstated.** On W2, eleven of those twelve
cells produce distinct trade sets; on W3, nine of twelve. Outside the acceptance
window the models genuinely do select different trades — they simply return
indistinguishably (all four configs land within ~0.5pp on W2). So the correct
claim is that the model is inert in its OUTCOME everywhere, and inert in its
MECHANISM on W1 specifically.

### Why adopt a change that does nothing

Three reasons, none of which is realised return on W1:

1. **It is already the configured value.** NNMT does not override
   `MIN_BUY_GAIN_THRESHOLD`, so 0.02 is what the next retrain of any NNMT model
   will use. Adopting it requires no action; rejecting it would require adding
   an NNMT-level override.
2. **It is the only one of the two that passes the standing oracle gate.** At
   0.007 the W1 guards-off ceiling is **−28.113%** (FAIL); at 0.02 it is
   **+17.791%** on 1561 trades (PASS). The old label describes trades that lose
   money even when predicted perfectly.
3. **It costs nothing measurable.** −0.157pp against 0.813pp of seed noise.

So the family keeps a target that *could* pay if the gate were ever opened, at
no measurable cost at the gate it actually runs. That is the whole of the case.

### FR-007 — models made consistent, or marked stale

The chosen label is 0.02. **Every pre-existing NNMT model on disk was trained at
0.007 and is therefore STALE with respect to the configured label:**

`NNMT_MLX` (Sep 3), `NNMT_DDPM_MLX`, `NNMT_CGP_MLX`, `NNMT_WGAN_MLX` (Sep 1),
and the ablation variants `NNMT_MLX_ctrl/noflow/nomom/norisk/profit/tronly/trprofit`
(Sep 1–2).

They are **not** retrained here, deliberately: this feature has just measured
that the label makes no difference to realised return at NNMT's gate, so forcing
eleven retrains would spend hours to reproduce the same numbers. They are marked
stale instead, and the next retrain of any of them picks up 0.02 automatically.
`NNMT_Lbl02_s42` and `NNMT_Lbl02_s7` are on-disk 0.02 models if one is wanted.

**The inconsistency that motivated this feature is therefore resolved as
BENIGN** — it was real, and it turned out not to matter.

---

## Change 2 — the profit-taking level, 0.032 → 0.022

### Verdict: **KEEP 0.022** (the inherited value). Indistinguishable from 0.032; not a defect.

Measured on the REAL model, never at the ceiling (R9: the oracle bounds what a
label can pay, not how the exit harvests it — demonstrated when this same change
was worth +0.91pp at the ceiling and 0.5–5.6pp on NNNC's real model).

| seed | window | 0.032 | 0.022 | Δ | trades differ |
|---|---|---|---|---|---|
| 42 | **W1** | +10.708% | +10.450% | **−0.258pp** | yes |
| 42 | W2 | +8.331% | +8.833% | +0.502pp | yes |
| 42 | W3 | +1.238% | +2.071% | +0.833pp | yes |
| 7 | **W1** | +10.708% | +10.450% | **−0.258pp** | yes |
| 7 | W2 | +8.724% | +9.261% | +0.537pp | yes |
| 7 | W3 | +1.238% | +2.072% | +0.834pp | yes |

Paired resolution on W1: −0.258pp against **2sd = 0.53pp** on 8 differing
trades, both seeds. **Within noise.**

The two seeds give *identical* W1 deltas with zero spread — not a coincidence,
and not independent confirmation either: both models produce the same W1 trade
set, so the exit change acts on identical trades. It is one measurement reported
twice, and is recorded that way rather than as two-seed agreement.

The sign flips between the acceptance window (−0.26pp) and both in-sample
windows (+0.50 to +0.83pp). Nothing is resolved in either direction.

**Keeping 0.022** because: it is not resolvably worse on W1; reverting would
require adding an NNMT-specific override for no measured gain; and the exit
config hash confirms the parameter is live rather than inert, so this is a
genuine null and not an absorbed change.

---

## What Phase 4 adds to the feature

The Phase 2 gate said a perfect entry signal is worth nothing here. Phase 4 shows
the same thing from the training side, and more sharply:

**The label change moved the guards-off oracle ceiling by 46pp — from a −28.1%
FAIL to a +17.8% PASS — and moved realised return by −0.16pp, which is noise.**

Two other metrics pointed the wrong way and neither was allowed a vote:

- `val_trading_mcc` said the OLD label was far better (0.760/0.758 vs
  0.612/0.609, consistent across both seeds). The 0.02 target is genuinely
  harder to learn — 2% in 48 bars against 0.7% — so a lower MCC is expected and
  is not evidence against it. It is recorded as context only (FR-019).
- The oracle ceiling said the NEW label was far better. Also correct, also
  irrelevant to what the strategy realises at this gate.

Both inherited changes are resolved, neither was a defect, and the family is
internally consistent. `val_profit_mcc ≈ 0.000` across all four models
independently confirms T013's static finding that the profit head is untrained.

---

## Addendum (2026-09-07) — re-testing the label with the SELL HEAD LIVE

**A gap in the Phase 4 reasoning, found and closed.** The label verdict above was
measured with `enable_exit_signal = False`, so the sell head never reached a
trade. But `MIN_SELL_LOSS_THRESHOLD` moved 0.007 → 0.02 as well, and the exit
promotion made the sell head load-bearing for the first time on this family.
"The label makes no difference" had therefore been established in a regime where
half of the label structurally could not matter.

Re-run: the same four models, exit signal ON at `exit_rvol_threshold` 1.0.

| window | 0.007 (s42 / s7) | 0.02 (s42 / s7) | label effect | seed spread |
|---|---|---|---|---|
| **W1** | +11.45 / +11.52 | +11.56 / +11.58 | **+0.086pp** | 0.077pp |
| W2 | +10.52 / +10.79 | +10.78 / +10.47 | −0.030pp | 0.302pp |
| W3 | +1.52 / +1.59 | +1.01 / +1.19 | −0.453pp | 0.183pp |

**The mechanism changed; the outcome did not.**

- With the exit signal OFF, the 0.02 and 0.007 arms produced **byte-identical**
  W1 trades — the label never reached the trades at all.
- With it ON, the trade hashes **differ in all three windows**. The label now
  genuinely reaches the trades, through the sell head.
- And it still buys nothing: **+0.086pp on W1 against a 0.077pp seed spread**.

So the Phase 4 verdict stands, now for a better reason. Previously the label was
unmeasurable (identical trades, INCONCLUSIVE under the standing protocol). Now it
is measurable and measured, and it is a null: the label reaches the trades and
changes essentially nothing.

W3's −0.453pp is the largest movement, and it is negative for the newer label.
W3 is in-sample and near-flat (R7) so it carries no acceptance weight, but it is
the second time in two days that W3 has disagreed with W1 on an exit-side change
(the other being the production take-profit interaction).

**Consequence for FR-007.** The eleven stale 0.007 models remain deliberately
un-retrained. That decision was made when the label was unmeasurable; it now
rests on a direct measurement in the regime the family actually runs in. The
staleness stays BENIGN.
