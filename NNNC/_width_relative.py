"""Width-relative labels for an NNNC model class (feature 033, contract C2).

The model check (research/model_calibration/findings.md) found that on the gbb rule's candidates an
NNNC model's ranking is largely a bb_width sort, and feature 032 showed that training on the
candidates alone does not remove it: the gbb_path label itself rewards wide bands (hit rate 0.45 ->
0.59 across width deciles). This mixin replaces the label with one that asks whether a bar did
better than bars of similar width (Framework/PooledLabel.py, applied in TrainingEngine):

  * entry side -- gbb-rule bars ranked by the gbb_path trade outcome within their width decile;
    the top half is BUY;
  * exit side  -- gbb_path_sell-rule bars ranked by the sell-path outcome within their own width
    deciles; the bottom half is SELL;
  * training rows -- either side's eligible bars with a valid outcome (`%wr_keep`), train and
    validation alike; deciles are fitted on the training rows only.

The label values come from the class's own label attributes, so the label differs from the
control's only in being width-relative. Entries are not changed here (the sizer arm uses the
width book's entries). The held-pair add is OFF, matching the comparators.
"""

GBB_ENTRY = (("guard_metric", "lt", 0.3), ("bb_width", "gt", 0.035))
GBB_EXIT = (("guard_metric", "gt", 0.3), ("bb_width", "gt", 0.035))


class WidthRelativeMixin:
    TRAIN_ROW_FILTER = (("%wr_keep", "gt", 0.5),)
    MODEL_KEY_ATTRIBUTES = ("TRAIN_ROW_FILTER", "POOLED_LABEL")
    HELD_PAIR_ADD = None
    WR_BUCKETS, WR_SHARE, WR_MIN_BUCKET = 10, 0.5, 500
    WR_STOP, WR_TAKE_PROFIT = -0.05, 0.022  # labels_gbb_path / labels_gbb_path_sell defaults
    MODEL_KEY_IGNORED = ("WR_BUCKETS", "WR_SHARE", "WR_MIN_BUCKET", "WR_STOP", "WR_TAKE_PROFIT")

    @property
    def POOLED_LABEL(self):  # computed from the class's own label attributes (model-key input)
        return {"target": float(self.MIN_BUY_GAIN_THRESHOLD), "stop": float(self.WR_STOP),
                "horizon": int(self.HORIZON), "reference": "bb_width",
                "eligible": GBB_ENTRY, "exit_eligible": GBB_EXIT,
                "exit_loss": float(self.MIN_SELL_LOSS_THRESHOLD),
                "exit_take_profit": float(self.WR_TAKE_PROFIT),
                "buckets": int(self.WR_BUCKETS), "share": float(self.WR_SHARE),
                "min_bucket": int(self.WR_MIN_BUCKET)}
