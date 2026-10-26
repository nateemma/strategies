# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""Entry veto on one score column (040 follow-up; research/nnmt/preregistration_own_trade_heads.json).

No entry on a bar whose VETO_COLUMN is below the fold's threshold (VETO_THRESHOLDS[MODEL_TRAIN_TIMERANGE]);
a missing score (NaN) vetoes too. Post-model: the trading model's weights are reused.

- VETO_MODEL None: the column is one of the strategy's own indicators (e.g. atr_pct_roll).
- VETO_MODEL "<prefix>": the column comes from a SECOND model run in-process on the same candles -- the class
  `<prefix>_<fold suffix>` of the same fold / seed (e.g. MT_LR_RI_AH_f03, MT_LR_RI_AH_f03_s2), or
  VETO_MAINLINE on a mainline class. It is built from this strategy's config and DataProvider, so backtest,
  dry and live compute it the same way. It never trains: its weights must exist (ModelKey raises otherwise
  in dry/live/plot; a backtest of a class without weights would train, so ensure_veto_weights() checks first).

VETO_THRESHOLDS None, or FT_VETO_OFF=1 in the environment (research/nnmt/head_veto.py pick only), switches the
veto off; the column is still computed. Thresholds are keyed "<train_timerange>" (seed 1) or
"<train_timerange>_s<seed>", "mainline" on a mainline class.
"""

import importlib
import os
import re

import numpy as np
import pandas as pd

RAW = ["date", "open", "high", "low", "close", "volume"]


class HeadVetoMixin:
    MODEL_KEY_IGNORED = ("VETO_COLUMN", "VETO_MODEL", "VETO_MAINLINE", "VETO_THRESHOLDS", "_veto_strat")
    MODEL_KEY_METHODS_IGNORED = ("get_entry_conditions", "_veto_score", "_veto_strategy", "_veto_class_name",
                                 "veto_threshold", "ensure_veto_weights")
    VETO_COLUMN: str = None
    VETO_MODEL: str = None        # fold-class prefix of the second model, or None (own column)
    VETO_MAINLINE: str = None     # the second model's mainline class (a class with no fold suffix)
    VETO_THRESHOLDS: dict = None  # {train_timerange: threshold}; None = veto off
    _veto_strat = None

    def _veto_class_name(self) -> str:
        m = re.search(r"_(f\d{2}(?:_s\d+)?)$", type(self).__name__)
        if m:
            return f"{self.VETO_MODEL}_{m.group(1)}"
        if self.VETO_MAINLINE is None:
            raise ValueError(f"{type(self).__name__}: no fold suffix and no VETO_MAINLINE")
        return self.VETO_MAINLINE

    def _veto_strategy(self):
        if self._veto_strat is None:
            name = self._veto_class_name()
            cls = getattr(importlib.import_module(name), name)
            s = cls(self.config)
            s.dp = self.dp
            s.dbg_verbose = False
            s.ft_load_hyper_params(False)
            from Framework import ModelKey as MK

            MK.locate_for(cls, {**self.config, "runmode": "plot"})  # plot mode: raises, never trains
            self._veto_strat = s
        return self._veto_strat

    def ensure_veto_weights(self) -> None:
        if self.VETO_MODEL is not None:
            self._veto_strategy()

    def _veto_score(self, dataframe) -> np.ndarray:
        if self.VETO_MODEL is None:
            return dataframe[self.VETO_COLUMN].to_numpy(float)
        s = self._veto_strategy()
        # mirror the host's run state (pickers / signal tables switch aggregation off); never train
        s.aggregate_pairs, s.training_needed = getattr(self, "aggregate_pairs", False), False
        meta = {"pair": self.curr_pair}
        # the model's predictions only (BaseNNStrategy.get_entry_conditions' two calls): none of the veto
        # model's post-model mixins (tr_scale, triggers, guards) run, so its class defaults cannot leak in
        other = s.advise_indicators(dataframe[RAW].copy(), meta)
        other = s.process_predictions(other, s.get_predictions(other, s.classifier))
        by_date = pd.Series(other[f"%{self.VETO_COLUMN}"].to_numpy(float), index=pd.to_datetime(other["date"], utc=True))
        return pd.to_datetime(dataframe["date"], utc=True).map(by_date).to_numpy(float)

    def veto_threshold(self):
        if self.VETO_THRESHOLDS is None or os.environ.get("FT_VETO_OFF") == "1":
            return None
        tr, seed = getattr(self, "MODEL_TRAIN_TIMERANGE", None), int(getattr(self, "MODEL_SEED", 1) or 1)
        key = "mainline" if not tr else tr if seed == 1 else f"{tr}_s{seed}"
        return self.VETO_THRESHOLDS[key]  # KeyError: a fold with no picked threshold

    def get_entry_conditions(self, dataframe):
        mask = super().get_entry_conditions(dataframe)
        score = self._veto_score(dataframe)
        dataframe["veto_score"] = score
        t = self.veto_threshold()
        if mask is None or t is None:
            return mask
        skip = ~(score >= t)  # NaN vetoes
        tags = getattr(self, "_st_tags", None)
        if tags is not None and len(tags) == len(dataframe):
            tags = tags.copy()
            tags[skip] = ""
            self._st_tags = tags
        return mask & ~pd.Series(skip, index=mask.index)
