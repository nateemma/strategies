# NNNC model check: what does the model select? (2026-10-13) — DIAGNOSTIC, PROVISIONAL

Inputs: [data.md](data.md) (calibration data); [within_candidates.py](within_candidates.py) →
`within_candidates.json`; [width_twin.py](width_twin.py) → `width_twin.json`;
[width_plus_model.py](width_plus_model.py) → `width_plus_model.json`. No training, no backtests:
the books are rebuilt with the NNNC framework study's simulator, which reproduces the recorded model
books at 99.7–100%. It has not been fidelity-checked on a non-model entry mask.

## 1. The high AUC is mostly the label's rule

- The gbb label is first a rule: `guard_metric < 0.3` and `bb_width > 0.035`. About 20% of bars
  pass it, and about half of those are labelled positive.
- Overall AUC is 0.93–0.98. On rule-eligible bars only, it falls to **0.60–0.62** (ARO, MLX),
  **0.58–0.59** (GBR) and **0.53–0.54** (GFM).
- The probability compression in data.md follows from this: label rates of 0.35–0.9 are squeezed
  into p 0.65–0.85.

## 2. Within the candidates, the model behaves like a band-width sort

The test holds selectivity fixed: the same number of rule-eligible bars as the model selects, chosen
by depth (`guard_metric`, declared), by width (`bb_width`, post-hoc) and at random. Mean 48-bar
forward return:

| model | half | model | depth | width (post-hoc) | random |
|---|---|---|---|---|---|
| ARO | early / late | +2.11 / +1.25 | +0.79 / +0.76 | +2.15 / +1.26 | +0.60 / +0.20 |
| GBR | early / late | +2.12 / +1.13 | +0.78 / +0.78 | +2.29 / +1.28 | +0.61 / +0.21 |
| GFM | early / late | +0.69 / +0.85 | +0.72 / +0.71 | +1.83 / +1.20 | +0.63 / +0.21 |
| MLX | early / late | +0.72 / +0.25 | +0.67 / +0.21 | +0.69 / +0.24 | +0.61 / +0.21 |

- **The declared rule passes, 3 of 4.** The model beats depth in both halves on ARO, GBR and MLX.
- **MLX's pass means little.** Its 0.69 threshold takes 80–93% of all eligible bars, so its
  selection is mostly the rule itself.
- **Against width (post-hoc), the model loses on ARO, GBR and GFM.** Its 48-bar +2.2% hit rate is
  a little higher (0.71–0.75 vs 0.67–0.71).

## 3. Book level: a model-free width book matches the model books

Built with the same gates and the gbb rule, then `bb_width ≥` a per-fold cut. The tradeable
variant uses the previous fold's cut. Simulator, 24 h grace, no held-pair add, pp early / late:

| | model book | width book (tradeable cut) | width AND model | width AND p ≥ 0.6 |
|---|---|---|---|---|
| ARO | 35.2 / 14.7 (607 / 569 tr) | 35.7 / 15.6 (412 / 519) | 19.4 / 13.5 (128 / 184) | 36.2 / 16.5 |
| GBR | 37.6 / 15.7 | 36.5 / 17.1 | 30.8 / 21.6 | 33.0 / 18.6 |
| GFM | 32.4 / 7.2 | **47.1 / 17.0** | 0.5 / 2.4 | 45.6 / 17.9 |
| MLX | 22.2 / 14.5 | 20.8 / 14.8 | 20.6 / 14.6 | 20.8 / 14.8 |

- **The width book alone matches ARO, GBR and MLX, and clearly beats GFM.**
- **Where the model agrees with width, it raises profit per trade:** ARO 0.15 vs 0.09 pp per trade,
  early. The cost is volume, and the 10-slot book is rarely full, so volume matters.

## Reading

This agrees with the 2026-09 guard ablation ("the model does not pick the trades"), now on the
guards-open, 24 h stack. It is also consistent with 023's finding that the model beats the plain
gbb trigger twin: the trigger twin did not sort by band width.
- **The model's measurable selection is close to one rule feature.** Its distinctive value is
  higher precision per trade, not more profit per book.
- **This does not say the model is useless.** It says the current label teaches mostly the rule,
  and what the model learns beyond the rule (within-candidate AUC about 0.6) is roughly what band
  width already supplies.

## Options (fix-oriented; the model stays)

1. **Make the width book the baseline that every model change must beat.** It costs nothing at run
   time, it is model-free, and it already exists in the simulator. A model variant that does not
   beat it adds nothing over one sorted feature.
2. **Re-target training at what the rule and width do not know.**
   - Train on rule-eligible bars only, so capacity goes to ranking candidates instead of learning
     the rule.
   - Optionally use a label defined relative to width: for example, forward outcome above the
     width-rank expectation.
   - The within-candidate AUC and the width-book comparison are the success measures, followed by
     walk-forward P&L.
3. **Use the model as a priority or sizing signal on the width book.** Agreement trades earn about
   1.7× per trade on ARO. For example, the width book's entries with a full stake when the model
   agrees and a smaller one when it doesn't. This is a cheap simulator screen.

**Before any of these:** confirm the width book with a real backtest. The simulator has not been
fidelity-checked on a non-model entry mask.
