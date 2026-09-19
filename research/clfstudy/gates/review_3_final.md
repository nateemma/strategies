# Gate-3 review, final (clean rows): phase 3 existing-type census

Classification metrics against fixed gbb labels (ind24, seq_len 16, windows A/B/C, production protocol lr 1e-4 / stride 1 / patience 10 / cap 40). Nothing here is evidence of trading edge (Constitution III). Sources: `results/results.jsonl` (phase-3 rows; clean = window A any date, B/C started >= 2026-09-18T07:00Z, last row per key: 95 rows, all `ok`), `gates/phase_3.json` (decided 2026-09-19 14:30Z), `report/methods.md` (ROOT CAUSE, trigger-rule ceiling, Mamba exclusion), `gates/review_4.md`. References: trigger rule A 0.673 / B 0.644 / C 0.642; clean RIDGE 0.533 / 0.535 / 0.547; fixed ROCKET 0.450 / 0.516 / 0.526; PERSISTENCE 0.80; MAJORITY 0. Supersedes `gates/review_3.md`.

## 1. What review_3.md got wrong

Every B/C row it read (seeds 0-1, phases 2-3) had been warm-started from the window-A checkpoint (methods.md ROOT CAUSE), including the RIDGE reference it quoted (0.307 / 0.299 on B/C; clean: 0.535 / 0.547). From that it inferred (a) B/C are a "harder regime" where neural types sit at 0.49-0.53 and beat RIDGE by 0.19-0.23; (b) 13 of 24 types fail the stability clause on B/C through early-stopping "loop noise", motivating a third-seed amendment; (c) dominance/drop calls on GRU, LSTM2, Mamba2, Multihead and the "TCN is the sweep most worth full seeds" anchor; (d) a cost projection built on those runs. All withdrawn. What survives: the window-A numbers (0.66 ceiling, LSTM_POOL -0.02, LSTM_INJIT -0.014, mean-pool readout matters), the MAJORITY/PERSISTENCE sanity checks, the empty-tail warning on buy_prec@0.8, and ROCKET's cost/seed-sensitivity note. The Multihead A gap it flagged was indeed contamination, but the clean re-run put Multihead at the ceiling, not below it.

## 2. Clean census (MCC mean +/- 2-seed std; pooled = mean over available windows; min/run = mean wall over clean rows)

| type | windows | pooled MCC / F1 | A | B | C | params | min/run |
|---|---|---|---|---|---|---|---|
| Transformer | ABC | 0.649 / 0.735 | 0.664 +/- .003 | 0.642 +/- .000 | 0.640 +/- .000 | 15,587 | 8.5 |
| LSTM (production) | ABC | 0.649 / 0.735 | 0.664 +/- .001 | 0.642 +/- .000 | 0.641 +/- .000 | 12,483 | 7.7 |
| LSTM2 | ABC | 0.649 / 0.735 | 0.663 +/- .000 | 0.643 +/- .000 | 0.640 +/- .000 | 86,787 | 14.3 |
| Multihead | ABC | 0.646 / 0.734 | 0.659 +/- .002 | 0.641 +/- .001 | 0.639 +/- .000 | 12,483 | 7.6 |
| GRU | ABC | 0.646 / 0.733 | 0.661 +/- .000 | 0.639 +/- .000 | 0.639 +/- .000 | 142,595 | 17.0 |
| Wavenet | ABC | 0.646 / 0.732 | 0.659 +/- .002 | 0.640 +/- .001 | 0.638 +/- .001 | 16,227 | 5.8 |
| Mamba2 | ABC | 0.640 / 0.729 | 0.644 +/- .011 | 0.639 +/- .001 | 0.638 +/- .001 | 19,407 | 10.5 |
| MLP | ABC | 0.640 / 0.727 | 0.653 +/- .001 | 0.634 +/- .002 | 0.633 +/- .002 | 136,515 | 3.8 |
| Attention | ABC | 0.612 / 0.706 | 0.618 +/- .005 | 0.601 +/- .001 | 0.618 +/- .001 | 12,483 | 6.5 |
| TSMamba | ABC | 0.592 / 0.707 | 0.502 +/- .004 | 0.638 +/- .002 | 0.636 +/- .002 | 37,435 | 15.3 |
| CNN | ABC | 0.549 / 0.663 | 0.555 +/- .004 | 0.537 +/- .004 | 0.555 +/- .003 | 2,211 | 2.6 |
| Mamba [A/B only] | AB | 0.492 / 0.601 | 0.513 +/- .079 | 0.470 +/- .019 | excluded (48 GB, 68 min) | 21,315 | 49.7 |
| LSTM_KAN [A + B seed 0] | A(B) | 0.663 / 0.743 (A) | 0.663 +/- .002 | 0.643 (n=1) | not run | 11,907 | 3.5 (A) |
| LSTM_PONDER [A only] | A | 0.665 / 0.745 | 0.665 +/- .001 | - | - | 14,659 | 4.2 |
| LSTM_NOISY / NORES [A only] | A | 0.664 / 0.744 | 0.664 +/- .000 | - | - | 12,483 | 3.6 |
| LSTM_D10 / NORESIZE / NOCONV [A only] | A | 0.663 / 0.743 | 0.663 +/- .000 | - | - | 12,483 / 10,539 / 12,483 | 3.6 |
| LSTM_K5 / W64 [A only] | A | 0.662 / 0.742 | 0.662 +/- .001 | - | - | 15,555 / 47,491 | 3.5 / 4.2 |
| LSTM_RAW [A only] | A | 0.661 / 0.741 | 0.661 +/- .001 | - | - | 10,539 | 3.5 |
| LSTM_INJIT [A only] | A | 0.650 / 0.739 | 0.650 +/- .001 | - | - | 12,483 | 3.6 |
| LSTM_POOL [A only] | A | 0.643 / 0.727 | 0.643 +/- .001 | - | - | 12,483 | 3.6 |

Sanity: no leakage (all < PERSISTENCE 0.80), no collapse (ceiling types predict ~11 / 73 / 16 % SELL/HOLD/BUY, BUY recall 1.0), every 2-seed std <= 0.011 except Mamba A (0.079). Per-window ranking among the top six is a tie: spread 0.005 on A, 0.004 on B, 0.003 on C, at seed std 0.000-0.003 - the order changes window to window and means nothing. B/C rows are 2-3x the wall time of A (larger windows); min/run above is the pooled mean, A-only entries are cheaper by construction. Thirteen pre-cutoff single-seed B/C rows (the 2026-09-17 seed-2 third seeds) remain in `leaderboard_3` / `phase_3.json` but are excluded here; 12 of 13 sit at 0.637-0.641 (agree with the ceiling), the 13th is LSTM_NORESIZE C/2 = 0.504 with best_epoch 6 - a from-scratch stall under the production protocol that the ceiling story does not explain and that the stop decision leaves untested.

## 3. Position against the trigger-rule ceiling (gap = type - rule, A / B / C)

- **At the ceiling (within 0.01 on every window):** Transformer -0.009 / -0.002 / -0.002; LSTM -0.009 / -0.002 / -0.001; LSTM2 -0.010 / -0.001 / -0.002. On A only (no clean B/C): PONDER, NOISY, NORES, D10, NORESIZE, NOCONV, KAN (-0.008 to -0.010).
- **Within 0.01 on B/C, 0.012-0.015 under on A:** Multihead -0.014 / -0.003 / -0.003; GRU -0.012 / -0.005 / -0.003; Wavenet -0.014 / -0.004 / -0.004; A-only K5, W64, RAW (-0.011 to -0.012). Everyone's A gap is ~0.005 wider than their B/C gap, so a strict 0.01-on-every-window rule splits this group from the first on A alone, by 0.003-0.005 at seed std 0.002 - real but immaterial; I treat both groups as ceiling-bound.
- **0.01-0.03 below:** Mamba2 -0.029 / -0.005 / -0.004 (A seed spread 0.011, one seed at the ceiling); MLP -0.020 / -0.010 / -0.009 (flattened 16x24 input, no last-bar readout); LSTM_INJIT -0.023 (A); LSTM_POOL -0.030 (A; mean-pool readout).
- **Structurally below** (hypotheses from the model code in `NNNC/NNNClassifierMLX.py`, not tested):
  - Attention -0.055 / -0.043 / -0.024 (12.5k params, same front-end as Multihead). `_AttentionModel` is one attention block with NO positional embedding, K/V projected from the raw input (bypassing the conv), post-LN, global average pool. Without position the block is permutation-invariant over the 16 bars and the mean pool cannot isolate the last bar, where the trigger (guard_metric, bb_width) is evaluated; `_MultiheadModel` adds `TransformerPositionalEmbeddingMLX` and self-attends on the conv output and reaches the ceiling. Phase 4's `pos=none` cost (-0.036 / -0.022 / -0.014, review_4 s.2) is the same effect at smaller size.
  - TSMamba -0.171 on A, but -0.007 / -0.006 on B/C (at the ceiling). Both A seeds stop at epochs 15-16 with best_epoch 4-5 at MCC 0.50 = RIDGE level (the linear half of the rule); on B/C best epochs 13-22. `_TSMambaModel` = bidirectional Mamba2 x 2 layers, averaged and mean-pooled, KAN head, RMSNorm. Hypothesis: an optimisation stall on the A distribution (4.0 / 6.3 % signal share) that patience 10 turns into an early stop, i.e. a protocol x architecture interaction, not a capacity limit; a patience-40 or lr 3e-4 re-run on A (2 x ~5 min) would settle it. Same signature as LSTM_NORESIZE C/2.
  - CNN -0.118 / -0.107 / -0.087 (2.2k params). `_CNNModel` = one causal conv, kernel 2 (receptive field 2 bars), tanh, BN, global average pool, 16-unit head. Receptive field 2 with no dilation and a mean pool that dilutes the last bar 1/16: it can only fit the linear approximation of the AND (RIDGE + 0.01). Phase 4's starvation axis (TCN ch16 or RF <= 5 costs 0.02-0.03; review_4 s.2) is the same effect; CNN is its extreme.
  - Mamba -0.160 / -0.174 (A / B), seed std 0.079 on A, logloss 0.81, ECE 0.16: never converges to the rule; excluded on C (methods.md 2026-09-19 05:15).

## 4. FR-028 as written (`gates/phase_3.json`: margin_used 0.020 = floor, RIDGE pooled seed std 0.002; std_bound 0.03 on every window; majority of 3; no failed/collapsed window)

- **Viable (10):** LSTM2, Multihead, Wavenet, MLP, Mamba2, Transformer, LSTM, Attention, TSMamba, GRU. Margins over RIDGE + 0.02: ceiling types clear by +0.07 to +0.11 MCC and +0.04 to +0.07 macro-F1 on every window; Attention by +0.05 / +0.05 / +0.05; TSMamba fails A (-0.051 below the bar) and passes on the B/C majority.
- **Fail (2 + 12):** CNN (A: MCC +0.001 over the bar but macro-F1 0.668 < 0.676; B/C fail both); Mamba (margin on both windows, std on A). The twelve LSTM ablations fail on `missing_window` (B/C retracted, not re-run) - an administrative fail, not a metric one; their A rows all pass the margin. LSTM_NORESIZE's "stability-only failure" in the JSON is a single-seed NaN-std artefact from the pre-cutoff rows, not instability.
- **Stability:** every clean 2-seed std is <= 0.011; 0 real stability failures. The third-seed amendment proposed in review_3.md is moot.
- **Non-discrimination:** the rule passes Transformer (0.664 / 0.642 / 0.640) and Attention (0.618 / 0.601 / 0.618) with the same verdict, and passes TSMamba although it sits at RIDGE level on one window. It is correct as a viability screen (everything neural except CNN/Mamba is >= 0.05 over RIDGE + margin) and has no resolution among ceiling-bound types (spread 0.005) - same conclusion as review_4 s.4. Amendment (record, do not bend): keep FR-028; add a "ceiling-bound" tag (within 0.01 of the trigger rule on B/C) and use the trigger-conditional metrics of s.5 as the selection metric.

## 5. Trigger-conditional metrics (phase-3 rows started after 2026-09-19 03:00Z: 42 rows, B/C only; `trig_rule_mcc` reproduces the rule at B 0.645 / C 0.642)

| type | BUY AUROC within trigger, B / C | argmax BUY precision lift, B / C | trig_buy_mcc B / C |
|---|---|---|---|
| Transformer | 0.590 / 0.623 | +0.000 / +0.000 | 0.01 / 0.01 |
| LSTM | 0.591 / 0.623 | +0.001 / +0.001 | 0.01 / 0.02 |
| LSTM2 | 0.588 / 0.623 | +0.000 / +0.000 | 0.01 / 0.01 |
| GRU / Wavenet / Mamba2 | 0.585 / 0.612, 0.584 / 0.614, 0.578 / 0.612 | +0.001 / +0.001 | 0.02 / 0.02 |
| MLP | 0.588 / 0.631 | +0.001 / +0.001 | 0.03 / 0.02 |
| Attention / TSMamba | 0.578 / 0.607, 0.561 / 0.599 | +0.000-0.003 | 0.00-0.03 |
| CNN | 0.571 / 0.588 | +0.013 / +0.017 | 0.07 / 0.09 |
| Mamba (B) | 0.551 | +0.002 | 0.02 |

Base rates 0.507 (B, n 35,867) / 0.454 (C, n 31,349). Reading: identical to phase 4 (review_4 s.5: B 0.566-0.592, C 0.600-0.626). Every ceiling-bound type orders triggered bars a little better than chance (AUROC ~0.59 B / ~0.62 C, seed std <= 0.008) and its argmax calls every triggered bar BUY (lift <= 0.001). Between-type AUROC differences are 0.01-0.02 on C at seed std up to 0.007: MLP's 0.631 is the highest single figure in either phase but not resolvable from Transformer/LSTM's 0.623. CNN's larger lift and trig_buy_mcc come from NOT reproducing the trigger (it withholds BUY on some triggered bars), not from more information - its AUROC is the lowest of the neural types. SELL side: AUROC 0.53-0.58, lift <= 0.006, for all types. Multihead and Mamba2 B/1 predate the instrument and carry no trig fields. Precision at reduced coverage (top-k by p_buy within the trigger) is still unmeasured - the only number that would show information beyond the trigger.

## 6. Gate-3 decision as it now stands

- **Finalist pool of existing types for any later work:** production **LSTM** (12,483 params; the deployment reference), **Multihead** (12,483; cheapest attention type), **Wavenet** (16,227; only conv type at the ceiling, cheapest per run at 5.8 min), **Transformer** (15,587; best pooled figure, +0.000 over LSTM). All four are within 0.003 of each other on B/C and within 0.005 on A, so the choice among them is cost, not metric; LSTM2 / GRU tie on metric at 7-11x the parameters and 2x the wall time and are dropped on cost. Mamba2 and MLP are 0.01-0.03 under and add nothing the four do not; Attention, TSMamba, CNN, Mamba are below the ceiling for the reasons in s.3.
- **Retracted from the earlier decision:** the third-seed amendment (moot - no clean stability failure exists); the 13 "stability-only" failures (warm-start artefacts); the "B/C regime is harder for neural types" reading (B/C are 0.03 harder for the RULE itself, 0.673 -> 0.644, and every type tracks it); the GRU/LSTM2 "dominated" calls now rest on cost only; the TCN-as-informative-sweep anchor (phase 4 showed the conv family reproduces Wavenet once the grid point is sane).
- **Standing conclusions consistent with review_4:** the architecture question is closed - every type that trains from scratch reaches the trigger rule and none exceeds it on any window; the only unmeasured quantity is within-trigger precision at reduced coverage, and phase 7 (inputs) is the only phase that can move it.
- **Not done, by author decision (recorded, not disputed):** clean B/C for the 12 LSTM ablations; Mamba on C; the TSMamba-on-A and LSTM_NORESIZE-C/2 stall diagnoses. None of these can change the finalist pool; the two stalls are the only loose thread in the "everything converges from scratch" claim.

Open question: does the author want the "ceiling-bound" tag and the trig-conditional selection metric written into `gates/phase_3.md` as an amendment, or left as a reviewer note here (review_4 Q1 asks the same for phase 4)?
