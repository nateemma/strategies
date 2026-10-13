# 031 T022 — fidelity: backtested adds vs the screen's adds (FR-009, SC-002)

Arm `hp_aro_dca` (NNNC_A_Raw_Open + `HELD_PAIR_ADD = {}`), f00–f07 (early half, already seen by the
screen; exempt in the pre-registration). Control: `wf_aro_g24`. Screen: `screen.py --dump-adds ARO
24` → `research/held_pair/adds_ARO.json` (DCA:all:1h at 24 h grace, 234 adds; the policy numbers
are unchanged from `screen_nnnc_r2.json`, +26.165 pp). Tool: `research/held_pair/fidelity.py` →
`research/held_pair/fidelity.json`.

| | value | gate |
|---|---|---|
| backtest adds | 239 | |
| screen adds | 234 | |
| matched on (pair, candle) | 233 | |
| **precision** (matched / backtest) | **97.5%** | ≥ 95% ✓ |
| recall (matched / screen) | 99.6% | reported |
| `row_misaligned` | 0 | 0 ✓ |
| exits on the add candle | 0 (0.00 pp) | reported |
| early-half gain, arm vs control | **+26.46 pp** | |
| early-half gain, screen | +26.17 pp | |

Per fold the arm and the screen agree within 0.1 pp, except f02 (+0.27 vs −0.17).

Six backtest adds have no screen add:
- LINK f01;
- BCH f02;
- AAVE, DOT and LINK on 2024-04-13 (the crash day, f05);
- LTC f07, one bar earlier than the screen's add.

These are cases where the screen's replayed trade had already exited, or was still too young,
at the moment the backtest's trade was still held. They are rebuild-path differences of a few
trades, not a timing or sizing defect.

**Result: PASS.** The backtest does what the screen priced, so the pre-registration may proceed.
