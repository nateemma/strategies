# Feature 003 — breakout tail labelling: consolidated findings

**Verdict: the label-specification hypothesis is REJECTED out-of-sample.**
The breakout labels were genuinely mis-specified relative to the risk taken,
the mis-specification was fixed, the fix reached the trades on every window,
and out-of-sample return did not improve.

Windows: W1 `20250901-20260831` (out-of-sample, the ONLY acceptance window),
W2 `20240601-20250531` and W3 `20230101-20231231` (both inside the
`20220101-20250831` training window, reported but excluded — FR-013).

## 1. Screening (no training, T011-T014)

    candidate            %bars  medGain  clears   r:r  peak(h)  stopped  status
    C0_h48_t005_CONTROL  9.07%     2.83   26.0%  0.57      6.8     4.5%  admitted
    C1_h48_t02           5.81%     4.29   40.6%  0.86      8.2     5.5%  admitted
    C2_h48_t03           4.30%     5.34   54.9%  1.07        -       -   EXCLUDED
    C3_h96_t03_TAIL      5.92%     6.41   66.8%  1.28     17.0    11.2%  admitted*
    C4_h96_t05           3.95%     8.43  100.0%  1.69     18.2    12.7%  EXCLUDED
    C5_h96_t03_MAE       5.25%     6.26   65.7%  1.25     16.8     0.0%  EXCLUDED
    C6_bb_tail           2.96%     6.54   67.1%  1.31     17.0    11.7%  EXCLUDED
    (* needs a hold-window change; density floor 5.5%)

Two candidates were killed for free, before any retrain:

- **C5, the MAE-constrained path-aware label (FR-006), WORKS** — it drives
  stopped-before-peak from 11.2% to **0.0%** — but pays 11% of the labels for
  it, dropping density to 5.25%, under the all-Hold collapse floor. It fails on
  COST, not on effect. Worth revisiting only at a setting with density to spare.
- **C6, the bb_position basis (FR-007), is simply too sparse** at tail settings
  (2.96%, roughly half the guard_metric basis).

Neither earned its additive labeler, so none was written into
`Framework/TrainingSignals.py`.

## 2. Every arm, W1 (out-of-sample)

    arm                          tr     ret%    DD%    PF   win%   verdict
    ctl003  h48/0.005, 12h       97    -0.74   3.74  0.94   58.8   (control)
    l1      h48/0.02,  12h       98    -0.20   3.80  0.98   60.2   unchanged
    h1      hold window ALONE    97    -1.09   3.98  0.91   56.7   unchanged
    l2      h96/0.03,  24h       98    -1.34   4.10  0.90   55.1   unchanged

Every arm changed the trades on all three windows — none was absorbed. The
power threshold is 1.0pp on 97 control trades (R9), so every delta above is
individually indistinguishable from noise.

## 3. Separated contributions (FR-011, SC-005)

    hold window alone             h1 - ctl003 = -0.35pp
    tail label GIVEN that window  l2 - h1     = -0.25pp
    -----------------------------------------------------
    package total                 l2 - ctl003 = -0.60pp

Valid because h1/l2 share a hold window and differ only in label, while
h1/ctl003 share a label and differ only in the hold window. `l1` is NOT
subtracted here — it carries a third label, so a three-way decomposition
against it would be arithmetically invalid.

## 4. The transferable finding

    arm       r:r   clears stop   W1 delta
    ctl003   0.57      26.0%       0.00pp
    l1       0.86      40.6%      +0.54pp
    l2       1.28      66.8%      -0.60pp

**Label economics do not predict return, and the relationship is
NON-MONOTONIC.** Reward:risk more than doubled and the share of labelled moves
able to clear the stop went from a quarter to two thirds; realised
out-of-sample return rose slightly at the intermediate cell and FELL at the
best-economics cell.

That makes reward:risk the fifth quantity in this study to fail as a proxy for
realised P&L, after EV-per-signal, precision, `val_mcc` and learnability MCC.
`val_mcc` failed again here: 0.5271 for `l2` against 0.5165 for `l1`, while
`l2` returned 1.14pp less.

**In-sample would have adopted a losing change.** `l2` improves on BOTH W2 and
W3 (+0.59pp each) while degrading W1; summed across all three windows it reads
**+0.58pp — a win.** Pinning W1 as the sole acceptance window is what caught it.

## 5. Harness caveat

`exit_config_hash` MATCHED between `h1` and the control despite a real
exit-behaviour change. It fingerprints exit PARAMETERS, and the hold window is
a method override, not a parameter. The TRADE hash caught it — same 97 trades,
different hash, meaning exit timing changed while entries did not. **For a
code-level exit change, trust the trade hash, not the exit-config hash.**

## 6. What is NOT established

- Whether the tail label ALONE would have worked. R4 declined to build that arm
  because its payoff peaks at 17.0h against a 12h forced exit, so it would be
  harvested early by construction — a false negative rather than a test. The
  consequence was accepted in advance.
- Whether a sub-1pp effect exists. 97 W1 trades cannot resolve half a point.
  Distinguishing one would need roughly an order of magnitude more trades than
  the guard chain admits — a statement about the arm's trade volume, not about
  the label.

## 7. Reproducing

    # screening (minutes, loads no model)
    PYTHONPATH=user_data/strategies .venv/bin/python \
      user_data/strategies/regime/screen_candidates.py

    # any arm, from the repo root
    PYTHONPATH=user_data/strategies .venv/bin/python \
      user_data/strategies/regime/run_arm.py --strategy <CLS> --window W1 --tag <tag>

    # the five-part comparison report
    PYTHONPATH=user_data/strategies .venv/bin/python \
      user_data/strategies/regime/tail/compare.py <arm_tag> ctl003

Arm records under `regime/arms/{ctl003,l1,h1,l2}_W{1,2,3}.json` are the durable
evidence. Control fingerprint: `exit_config_hash 203297442969c7da`.
