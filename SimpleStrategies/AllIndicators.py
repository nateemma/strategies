"""AllIndicators (feature 037, research R14): every SimpleStrategies indicator and raw signal on one
chart. It NEVER trades -- populate_entry_trend / populate_exit_trend set no signals.

populate_indicators runs the 037 harvest (research/indicator_screen/harvest.py) for the pair: the
base columns as `base:<col>`, each strategy's columns as `<Class>:<col>`, and raw (pre-filter)
signals as `raw_buy:<Class>[:variant]` / `raw_sell:...`. The screens read the cache, not this class.

    freqtrade plot-dataframe -s AllIndicators --strategy-path user_data/strategies/SimpleStrategies --no-trades -c user_data/strategies/config/config_wf.json \
        -p SOL/USDT --timerange 20250101-20250201 --indicators2 raw_buy:CCI CCI:cci
"""
import sys
from pathlib import Path

import pandas as pd
from pandas import DataFrame

group_dir = str(Path(__file__).parent)
strat_dir = str(Path(__file__).parent.parent)
sys.path.append(strat_dir)
sys.path.append(group_dir)

from SimpleStrategy import SimpleStrategy  # noqa: E402


class AllIndicators(SimpleStrategy):

    plot_config = {"main_plot": {"close": {"color": "lightsteelblue"}}, "subplots": {}}

    _harvest_strats = None

    def _strats(self):
        if self._harvest_strats is None:
            from research.indicator_screen import discover as D
            from research.indicator_screen import harvest as H

            self._harvest_strats = [H.load_strategy(self.config, c["class"])
                                    for c in D.discover()["classes"]]
        return self._harvest_strats

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        from research.indicator_screen import harvest as H

        frame, _ = H.harvest_frame(dataframe[["date", *H.OHLCV]].reset_index(drop=True),
                                   self._strats(), None, metadata["pair"])
        extra = frame.drop(columns=["date", *H.OHLCV])
        extra.index = dataframe.index
        return pd.concat([dataframe, extra], axis=1)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = 0
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        return dataframe
