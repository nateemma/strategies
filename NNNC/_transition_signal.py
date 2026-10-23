# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301
# type: ignore
"""040 transition arms (research/nnmt/preregistration_transition.json "arms"; T044).

The stand-in's per-fold predictions (research/nnmt/transition.py screen ->
user_data/data/nnmt_transition/pred_fNN.parquet: date, pair, cell, part tail/test, p) drive four
post-model switches. A threshold is a percentile of that fold's TAIL predictions for the cell (pooled
over pairs; the tail ends H bars before test_start, so it never sees the test window). The prediction at
candle t uses inputs up to t's close, so acting on it at t's signal is causal. A bar with no prediction
row (not tradeable) never fires.

  TrGateMixin   cs_aro's book skips an entry while P(DOWN) >= tail p80
  TrScaleMixin  cs_aro's stake x1.333 when P(UP) >= tail p80, x0.667 when P(DOWN) >= tail p80 (both:
                unchanged); a side whose class selected no cell stays unscaled
  TrExitMixin   cs_aro closes an open trade at the first candle (at or after the entry candle) with
                P(DOWN) >= tail p90 -- via custom_exit, which exit_profit_only does not block
  TrLongMixin   standalone long entry when P(UP) >= tail p99 (SimpleTrigger, st_g24_rev's exit stack:
                tagged reversion:tr_up, so the reversion family exits apply; no trigger, no regime gate)

The cells (TR_UP_CELL / TR_DOWN_CELL) are written by research/nnmt/build_arms.py transition from
screens_transition.json "selected". Every attribute and method here is post-model: the cs_aro arms keep
cs_aro's weights (the builder asserts the model key). The fold id comes from the class name.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

TABLES = Path(__file__).resolve().parents[2] / "data" / "nnmt_transition"


class TransitionSignalMixin:
    TR_UP_CELL = None
    TR_DOWN_CELL = None
    TR_TABLES = TABLES  # the seed-2 arms point at nnmt_transition_rs1/
    MODEL_KEY_IGNORED = ("TR_UP_CELL", "TR_DOWN_CELL", "TR_TABLES", "TR_GATE_Q", "TR_SCALE_Q", "TR_SCALE_WEIGHTS",
                         "TR_EXIT_Q", "TR_LONG_Q")
    MODEL_KEY_METHODS_IGNORED = ("_tr_fold", "_tr_cell", "_tr_threshold", "_tr_p", "_tr_fires")

    def _tr_fold(self) -> str:
        m = re.search(r"_(f\d{2})(?:_|$)", type(self).__name__)
        if not m:
            raise ValueError(f"{type(self).__name__}: no fold id in the class name")
        return m.group(1)

    def _tr_cell(self, cell: str) -> pd.DataFrame:
        cache = self.__dict__.setdefault("_tr_cache", {})
        if cell not in cache:
            t = pd.read_parquet(Path(self.TR_TABLES) / f"pred_{self._tr_fold()}.parquet")
            t = t[t["cell"] == cell]
            if t.empty:
                raise ValueError(f"{type(self).__name__}: no predictions for cell {cell!r}")
            t = t.assign(date=pd.to_datetime(t["date"], utc=True))
            cache[cell] = t
        return cache[cell]

    def _tr_threshold(self, cell: str, q: float) -> float:
        tail = self._tr_cell(cell)
        tail = tail.loc[tail["part"] == "tail", "p"].to_numpy(float)
        if not len(tail):
            raise ValueError(f"{type(self).__name__}: no tail predictions for {cell!r}")
        return float(np.percentile(tail, q))

    def _tr_p(self, cell: str, pair: str, dates) -> np.ndarray:
        """P(cell) per date for `pair` (NaN where the bar has no prediction)."""
        t = self._tr_cell(cell)
        s = t[(t["part"] == "test") & (t["pair"] == pair)].set_index("date")["p"]
        return pd.to_datetime(pd.Series(dates), utc=True).map(s).to_numpy(float)

    def _tr_fires(self, cell, q: float, pair: str, dates) -> np.ndarray:
        if cell is None:
            return np.zeros(len(dates), dtype=bool)
        p = self._tr_p(cell, pair, dates)
        return np.nan_to_num(p, nan=-np.inf) >= self._tr_threshold(cell, q)


class TrGateMixin(TransitionSignalMixin):
    TR_GATE_Q = 80.0

    def get_entry_conditions(self, dataframe):
        mask = super().get_entry_conditions(dataframe)
        if mask is None:
            return mask
        if self.TR_DOWN_CELL is None:
            raise ValueError(f"{type(self).__name__}: tr_gate needs a selected DOWN cell")
        skip = self._tr_fires(self.TR_DOWN_CELL, self.TR_GATE_Q, self.curr_pair, dataframe["date"])
        return mask & ~pd.Series(skip, index=mask.index)


class TrScaleMixin(TransitionSignalMixin):
    TR_SCALE_Q = 80.0
    TR_SCALE_WEIGHTS = (4.0 / 3.0, 2.0 / 3.0)  # (P(UP) high, P(DOWN) high)
    _SUFFIX = {"up": "|tr_up", "down": "|tr_down"}

    def get_entry_conditions(self, dataframe):
        mask = super().get_entry_conditions(dataframe)
        tags = getattr(self, "_st_tags", None)
        if mask is None or tags is None:
            raise ValueError(f"{type(self).__name__}: tr_scale needs the book's tagged mask")
        d, pair = dataframe["date"], self.curr_pair
        up = self._tr_fires(self.TR_UP_CELL, self.TR_SCALE_Q, pair, d)
        down = self._tr_fires(self.TR_DOWN_CELL, self.TR_SCALE_Q, pair, d)
        m = np.asarray(mask, dtype=bool)
        tags = tags.copy()
        for sel, suf in ((m & up & ~down, self._SUFFIX["up"]), (m & down & ~up, self._SUFFIX["down"])):
            tags[sel] = [f"{t}{suf}" for t in tags[sel]]
        self._st_tags = tags
        return mask

    def stake_weight(self, entry_tag) -> float:
        tag = str(entry_tag or "")
        for side, suf in self._SUFFIX.items():
            if tag.endswith(suf):
                w = self.TR_SCALE_WEIGHTS[0 if side == "up" else 1]
                return super().stake_weight(tag[: -len(suf)]) * w
        return super().stake_weight(entry_tag)


class TrExitMixin(TransitionSignalMixin):
    TR_EXIT_Q = 90.0

    def custom_exit(self, pair, trade, current_time, current_rate, current_profit, **kwargs):
        if self.TR_DOWN_CELL is None:
            raise ValueError(f"{type(self).__name__}: tr_exit needs a selected DOWN cell")
        df, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        if not df.empty:
            last = pd.Timestamp(df["date"].iloc[-1])
            if last >= pd.Timestamp(trade.open_date_utc) and \
                    self._tr_fires(self.TR_DOWN_CELL, self.TR_EXIT_Q, pair, [last])[0]:
                return "tr_exit"
        return super().custom_exit(pair, trade, current_time, current_rate, current_profit, **kwargs)


class TrLongMixin(TransitionSignalMixin):
    TR_LONG_Q = 99.0
    TR_LONG_TAG = "reversion:tr_up"

    def get_entry_conditions(self, dataframe):
        if self.TR_UP_CELL is None:
            raise ValueError(f"{type(self).__name__}: tr_long needs a selected UP cell")
        m = self._tr_fires(self.TR_UP_CELL, self.TR_LONG_Q, self.curr_pair, dataframe["date"])
        tags = np.full(len(dataframe), "", dtype=object)
        tags[m] = self.TR_LONG_TAG
        self._st_tags = tags
        return pd.Series(m, index=dataframe.index)
