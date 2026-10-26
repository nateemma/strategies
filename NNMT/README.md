# NNMT — Neural Network Multi-Task Classification Strategies

Multi-task classifiers — instead of predicting only the trading action,
the model predicts six related labels at once: trading, regime, risk,
momentum, flow, profit.  Forcing the network to balance across multiple
heads acts as a regulariser and produces a model that's less prone to
overfitting any single task.

Inherits from `BaseNNStrategy` (`Framework/BaseNNStrategy.py`) via `NNMTStrategy`
(`NNMT/NNMTStrategy.py`): `NNMTStrategy` → `BaseNNStrategy`. The former `BaseNNMTStrategy` was
merged into `NNMTStrategy` (2026-06-12), which now holds the multi-task scaffolding (target
calculators, pipeline overrides, task filtering, multi-task class attributes) as well as the
strategy-specific parts (plot config, classifier choice, diagnostics, profit-scaling helpers).

## Status (feature 040, 2026-10-24; PROVISIONAL)

Full results: `research/nnmt/report.md`.

- **Framework parity.** `NNMTStrategy` carries `EntryFrameworkMixin` (`ENTRY_MODE = "model"`,
  `MODEL_ENTRY_FAMILY = None`, `HELD_PAIR_ADD = None`); trade-identical to the pre-port NNMT on 15/15
  folds. Every NNNC switch is declarable but off.
- **As it stands NNMT is guard-decided:** model-decided share 0.001 (21.00 pp over the 15 folds vs
  `cs_aro` 59.95). At the operating point (guards open) the model decides (share 0.84) but the late half
  loses (−18 pp).
- **Model variants:** `NNMT_LR_RI` (plain LSTM on raw8 + the 24 indicators, `_nnmt_inputs.py`, at ARO's
  operating point) is the selected base, confirmed at seed 2 but only "no worse, slightly better" than the
  old network (now `NNMT_MLX_Conv`) at the operating point. Raw-only inputs lose ~20 pp early.
- **`NNMT_MLX` IS `NNMT_LR_RI` since 2026-10-26** (user decision; same model key, shared weights; also much
  faster to train). The pre-2026-10-26 network (Conv preamble + LSTM, LSTM head per task, NNMT guards) is
  `NNMT_MLX_Conv`; every class and record that used it now inherits it from there (keys unchanged).
- **Defaults adopted 2026-10-24 (on `NNMTStrategy`):** 24 h stop-loss grace (`mt_lr_ri_g24`, PASS both
  seeds, 40.9 / 42.2) and the `tr_scale` exposure scaler (PASS both seeds); 040 records pin the old
  values (`_pre_g24.py`). Held-pair add, gbb gate (near-inert here), C2 take-profit FAIL.
- **Auxiliary heads: don't use.** Each beats its persistence baseline, none ranks width-book trades; a
  trading-head-only twin shows no evidence multi-task training helps. The `p_buy` sizer on `w_aro`'s book
  passes at seed 1 and fails at seed 2.
- **Regime transition:** a stand-in model (not an NNMT head; the NNMT head does not match it) used as an
  exposure scaler on `cs_aro` (`tr_scale`) passes both seeds; a standalone long (`tr_long`) is positive
  both seeds. Gate and exit uses fail.

## Main files

| File | What it does |
|---|---|
| `NNMTStrategy.py` | Family base class.  Builds the multi-task label dict (`{"trading": …, "regime": …, …}`) and wires the multi-task classifier into the standard NN pipeline.  Tabular multi-task GAN augmentation (e.g. `MT_CTAB_GAN`) flows through `BaseNNStrategy.enhance_training_data` — set `gan_type = GANType.MT_CTAB_GAN` on the subclass and the dispatcher routes through `GANs.balance.balance_multi_task` automatically. |
| `NNMTClassifier.py` | TF/Keras multi-task classifier factory.  `ClassifierType` enum + `create_classifier()`; 14 architecture variants split into "Normal" (LSTM, Transformer, CNN, GRU, Wavenet, Wavenet_Fast, Attention) which override the shared backbone, and "Multi_*" variants which override every per-task head. |
| `NNMTClassifierMLX.py` | Apple-MLX multi-task classifier factory.  `ClassifierTypeMLX` enum + `create_classifier_mlx()`.  Mirrors the TF version's 14 architectures, but every Multi_\* variant uses the same head architecture for all six tasks (no per-task branching). |
| `NNMT_CGP.py`, `NNMT_CGP_Attention.py` | Concrete strategies using CTAB-GAN+ multi-task augmentation. |
| `NNMT_WGAN.py`, `NNMT_WGAN_MLX.py` | Concrete strategies using multi-task WGAN-GP augmentation. |

## Multi-task label format

The trainer expects labels as a dict, not a single one-hot array:

```python
{
    "trading":  one_hot,  # (N, 3) sell/hold/buy
    "regime":   one_hot,  # (N, 3) bear/sideways/bull
    "risk":     one_hot,  # (N, 3) low/normal/high
    "momentum": one_hot,  # (N, 3) negative/stable/positive
    "flow":     one_hot,  # (N, 3) decrease/neutral/increase
    "profit":   one_hot,  # (N, 3) loss/neutral/profit
}
```

This is what the classifiers (`NNMTClassifier` / `NNMTClassifierMLX`)
and the multi-task GANs (`MT_WGAN`, `MT_CTAB_GAN`) all consume.

## Adding a new variant

1. Create `NNMT_<Name>.py` here.
2. Inherit from `NNMTStrategy` (or one of the `NNMT_CGP*` / `NNMT_WGAN*`
   shims if the augmentation is the same).
3. Override `get_classifier_type()` to pick a `ClassifierType` /
   `ClassifierTypeMLX` value.
4. Run a long-timerange backtest to train and save.
