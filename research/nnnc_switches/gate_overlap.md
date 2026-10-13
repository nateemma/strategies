# Why the volreg gate hurts NNNC: overlap diagnosis (2026-10-12)

*PROVISIONAL. No backtests.* Run on the switch experiment's records with
`research/nnnc_switches/gate_overlap.py` (data: `gate_overlap.json`). Off state = volreg < 1.3658
at the signal bar.

## 1. Overlap: the models lean away from the off state, but are far from avoiding it

| rows | off-state share |
|---|---|
| all bars | 0.639 |
| gbb trigger rows (`st_book_rev2_b029`, before its gate) | 0.519 |
| arm D model signals | **0.348** |
| GbbRaw model signals | 0.379 |
| GbbFwdMae model signals | 0.425 |
| NNNC_MLX model signals | 0.503 |

The models fire in the off state less often than the trigger does. Arm D's share is a third lower.
But 35–50% of their signals are still off-state, so the gate removes a large share of trades
(37–51%).

## 2. Buckets: the model's off-state trades make money; the trigger's lose

Trades of the ungated arm (gate-off control for each model; `st_gbb_b029` for the trigger), split
by state.

| | early on | early off | late on | late off |
|---|---|---|---|---|
| **gbb trigger** | 1843: +42.39 (2.30) | 3184: **−12.07** (−0.38) | 1167: +13.29 (1.14) | 2862: **−32.57** (−1.14) |
| **arm D** | 444: +24.82 (5.59) | 280: **+4.42** (1.58) | 377: +6.30 (1.67) | 321: **+6.56** (2.04) |
| **GbbRaw** | 667: +33.88 (5.08) | 599: −4.53 (−0.76) | 485: +10.10 (2.08) | 641: +4.39 (0.68) |
| **GbbFwdMae** | 620: +13.87 (2.24) | 609: +2.83 (0.46) | 377: +7.84 (2.08) | 438: +0.69 (0.16) |
| **NNNC_MLX** | 127: +13.49 (10.62) | 69: +3.53 (5.12) | 105: +6.86 (6.53) | 76: +1.29 (1.70) |

Each cell is: trades: Σ pp (mean bp of wallet per trade).

- **The gate's premise holds for the trigger.** Off-state gbb trades lose in both halves, which is
  why 026 adopted it there.
- **It does not hold for the model's selection.** Off-state model trades are positive in 7 of 8
  model-half cells. The one exception is GbbRaw early, and that is the one cell where the gate
  gained (+5.44).
- **The model has largely already removed the off-state losers.** Off-state trades are usually
  weaker per trade than on-state ones, but they still make money. Arm D's late off-state trades
  are even slightly better than its on-state ones.

## 3. Decomposition: the loss is the removed trades, not substitution

The gate's effect splits exactly into three parts: gate − gate-off = removed − displaced + added.

| model | gate − gate-off (pp) | removed (−P&L of off trades) | displaced | added |
|---|---|---|---|---|
| arm D | −7.02 | −10.98 (601 trades) | −3.10 (7) | +0.86 (34) |
| GbbRaw | −2.41 | +0.14 (1240) | −4.01 (27) | −6.52 (121) |
| GbbFwdMae | −3.29 | −3.52 (1047) | +0.21 (1) | +0.43 (13) |
| NNNC_MLX | −5.01 | −4.83 (145) | +0.09 (1) | −0.08 (7) |

The columns are:
- **displaced:** P&L of the control's on-state trades that the gate arm didn't take;
- **added:** P&L of new trades in the gate arm.

No gate-arm trade enters in the off state, so the gate behaves as declared.

- **Arm D, GbbFwdMae and NNNC_MLX:** almost all the loss is the profitable off-state trades the
  gate removes.
- **GbbRaw:** its removed trades net out to about zero (−4.53 early, +4.39 late). Its loss comes
  from trades the gate arm takes instead, mostly in the late half (−8.72).

## Conclusion

The volreg gate removes reversion entries that lose *on average across the trigger*. The model
already performs much of that selection: in the off state it keeps entries that make money. A
hard gate on top therefore removes good trades. This is consistent with 029's finding that arm D's
model is selection *within* the gbb trigger. An entry filter that works on SimpleTrigger targets
the entries the model already rejects, so it does not carry over to NNNC. The exit switch (C2)
partly carried over: 2 of 4 models.

## Fix-oriented options (not run)

1. **A soft gate.** In the off state, raise the model's prediction threshold rather than refusing
   outright, keeping only high-confidence off-state signals. Off-state trades are usually weaker
   per trade, so a margin-based cut targets the residual. It needs a small framework switch (a
   per-state threshold offset) and a pre-registered walk-forward arm.
2. **volreg as a model input.** volreg is cross-sectional (a median over 10 pairs), so it carries
   information a per-pair model doesn't currently see. Testing it needs retraining.
3. **Use SimpleTrigger for exits; use model-mode controls for entry filters.** Test entry-side
   framework features on a model-mode control from the start, rather than carrying them over from
   trigger results.
