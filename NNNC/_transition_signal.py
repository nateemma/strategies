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

TR_SOURCE (2026-10-24): "tables" (default; every 040 record ran it) reads the per-fold stand-in tables;
"standin" scores the fitted stand-in INSIDE the strategy (_transition_standin.py), the same code in a
backtest and a live bot: BTC/USDT is an informative pair, the long-window inputs come from
utils/CloseHistoryStore (local files + the bot's candles; persisted in dry-run/live), the cutoff is
TR_STANDIN_CUTOFF or the MODEL_TRAIN_TIMERANGE end, and the thresholds are that fit's tail percentiles.
Where the inputs are unavailable the probability is NaN, so the scaler leaves the stake unchanged.
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
    TR_SOURCE = "tables"  # or "standin" (live-capable)
    TR_STANDIN_CUTOFF = None  # YYYYMMDD; None = the MODEL_TRAIN_TIMERANGE end
    MODEL_KEY_IGNORED = ("TR_UP_CELL", "TR_DOWN_CELL", "TR_TABLES", "TR_GATE_Q", "TR_SCALE_Q", "TR_SCALE_WEIGHTS",
                         "TR_EXIT_Q", "TR_LONG_Q", "TR_SOURCE", "TR_STANDIN_CUTOFF")
    MODEL_KEY_METHODS_IGNORED = ("_tr_fold", "_tr_cell", "_tr_threshold", "_tr_p", "_tr_fires", "_tr_fires_df",
                                 "_tr_standin", "_tr_store", "_tr_probs", "_tr_cutoff", "informative_pairs")

    # ------------------------------------------------------------------ stand-in source (live-capable)
    def _tr_cutoff(self) -> str:
        if self.TR_STANDIN_CUTOFF:
            return str(self.TR_STANDIN_CUTOFF)
        tr = getattr(self, "MODEL_TRAIN_TIMERANGE", None)
        if not tr:
            raise ValueError(f"{type(self).__name__}: TR_SOURCE standin needs TR_STANDIN_CUTOFF or MODEL_TRAIN_TIMERANGE")
        return tr.split("-")[1]

    def _tr_standin(self):
        st = self.__dict__.get("_tr_standin_obj")
        if st is None:
            from _transition_standin import StandIn

            st = self.__dict__["_tr_standin_obj"] = StandIn(self._tr_cutoff())
        return st

    def _tr_store(self):
        store = self.__dict__.get("_tr_store_obj")
        if store is None:
            from utils.CloseHistoryStore import CloseHistoryStore

            mode = str(getattr(self.config.get("runmode", ""), "value", self.config.get("runmode", "")))
            live = mode in ("live", "dry_run")
            persist = Path(self.config["user_data_dir"]) / "data" / "transition_history" / type(self).__name__ if live else None
            store = self.__dict__["_tr_store_obj"] = CloseHistoryStore(
                self.config["datadir"], self.timeframe, self.config.get("dataformat_ohlcv", "feather"),
                persist_dir=persist, keep_bars=20000 if live else None)
        return store

    def _tr_probs(self, dataframe: pd.DataFrame) -> dict:
        """{cell: P aligned with dataframe} from the in-strategy stand-in (NaN where not tradeable/unavailable)."""
        import _transition_standin as TS

        pair = self.curr_pair
        key = (pair, len(dataframe), pd.Timestamp(dataframe["date"].iloc[-1]))
        cache = self.__dict__.setdefault("_tr_prob_cache", {})
        if key in cache:
            return cache[key]
        store = self._tr_store()
        store.update(pair, dataframe)
        btc = self.dp.get_pair_dataframe(TS.CONTEXT_PAIR, self.timeframe)
        if btc is not None and len(btc):
            store.update(TS.CONTEXT_PAIR, btc)
        first = pd.Timestamp(dataframe["date"].iloc[0])
        hp, hb = store.series(pair, first), store.series(TS.CONTEXT_PAIR, first)
        cells = [c for c in (self.TR_UP_CELL, self.TR_DOWN_CELL) if c]
        if hp is None or hb is None or btc is None or not len(btc):
            out = {c: np.full(len(dataframe), np.nan) for c in cells}
            if not self.__dict__.get("_tr_warned"):
                self.__dict__["_tr_warned"] = True
                import logging

                logging.getLogger(__name__).warning(
                    "%s: transition inputs unavailable for %s (history store / BTC); tr signal neutral",
                    type(self).__name__, pair)
        else:
            X, tradeable = TS.features(dataframe, btc, hp, hb)
            st = self._tr_standin()
            out = {c: st.predict(c, X, tradeable) for c in cells}
        cache.clear()
        cache[key] = out
        return out

    def _tr_fires_df(self, cell, q: float, dataframe: pd.DataFrame) -> np.ndarray:
        """_tr_fires for this dataframe, from TR_SOURCE."""
        if cell is None:
            return np.zeros(len(dataframe), dtype=bool)
        if self.TR_SOURCE == "tables":
            return self._tr_fires(cell, q, self.curr_pair, dataframe["date"])
        if self.TR_SOURCE != "standin":
            raise ValueError(f"{type(self).__name__}: TR_SOURCE {self.TR_SOURCE!r}")
        p = self._tr_probs(dataframe)[cell]
        return np.nan_to_num(p, nan=-np.inf) >= self._tr_standin().threshold(cell, q)

    def informative_pairs(self):
        pairs = list(super().informative_pairs())
        if self.TR_SOURCE == "standin":
            from _transition_standin import CONTEXT_PAIR

            if (CONTEXT_PAIR, self.timeframe) not in pairs:
                pairs.append((CONTEXT_PAIR, self.timeframe))
        return pairs

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
        skip = self._tr_fires_df(self.TR_DOWN_CELL, self.TR_GATE_Q, dataframe)
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
        up = self._tr_fires_df(self.TR_UP_CELL, self.TR_SCALE_Q, dataframe)
        down = self._tr_fires_df(self.TR_DOWN_CELL, self.TR_SCALE_Q, dataframe)
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
            if last >= pd.Timestamp(trade.open_date_utc) and (
                    self._tr_fires(self.TR_DOWN_CELL, self.TR_EXIT_Q, pair, [last])[0] if self.TR_SOURCE == "tables"
                    else self._tr_fires_df(self.TR_DOWN_CELL, self.TR_EXIT_Q, df)[-1]):
                return "tr_exit"
        return super().custom_exit(pair, trade, current_time, current_rate, current_profit, **kwargs)


class TrLongMixin(TransitionSignalMixin):
    TR_LONG_Q = 99.0
    TR_LONG_TAG = "reversion:tr_up"

    def get_entry_conditions(self, dataframe):
        if self.TR_UP_CELL is None:
            raise ValueError(f"{type(self).__name__}: tr_long needs a selected UP cell")
        m = self._tr_fires_df(self.TR_UP_CELL, self.TR_LONG_Q, dataframe)
        tags = np.full(len(dataframe), "", dtype=object)
        tags[m] = self.TR_LONG_TAG
        self._st_tags = tags
        return pd.Series(m, index=dataframe.index)
