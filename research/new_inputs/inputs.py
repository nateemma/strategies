"""035 T003: the declared input table (research R6), the single source for every 035 script.

17 inputs in four families. Each is computed at the bar's close from trailing data only and
normalised without a fitted artefact (R1):

  z  -- (x - trailing 2880 mean) / trailing 2880 std, min_periods 960, clipped to +-5, / 5
  c  -- 2s - 1 for a share s in [0, 1]

Windows (bars of 15 m): 4 = 1 h, 16 = 4 h, 96 = 24 h, 2880 = 30 d. Declared once, never tuned.
Positioning inputs carry a missing flag (clarification Q3): 1 wherever the normalised value is
undefined (raw absent, or z warm-up).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field


Z_WINDOW, Z_MIN, Z_CLIP = 2880, 960, 5.0
SNAPSHOT_DELAY_MIN = 5          # a 5-min metrics snapshot is usable 5 min after its stamp (R3)
SNAPSHOT_STALE_MIN = 60         # an as-of snapshot older than this at the bar close is absent
MIN_PAIRS = 6                   # market state needs at least this many clean pairs (026)

#: the 10 config_wf pairs (the model's universe), and the market-context series
PAIRS = ("ZEC/USDT", "XRP/USDT", "SOL/USDT", "LINK/USDT", "NEAR/USDT", "AAVE/USDT",
         "AVAX/USDT", "LTC/USDT", "BCH/USDT", "DOT/USDT")
CONTEXT = "BTC/USDT"


@dataclass(frozen=True)
class Input:
    tag: str
    family: str           # order_flow | positioning | market_state | basis
    source: str           # global_spot | us_spot | um_metrics | um_premium | binanceus_ohlcv
    raw: str              # builder function name in build.py
    norm: str             # z | c
    windows: dict = field(default_factory=dict)
    missing_flag: bool = False
    live_source: str = ""


TABLE = (
    Input("OF_TBG", "order_flow", "global_spot", "taker_share", "c", {"sum": 4}, False,
          "none: api.binance.com 451 from here; archive lags a day"),
    Input("OF_TRG", "order_flow", "global_spot", "trade_count_ratio", "z", {"sum": 4, "median": 96},
          False, "none: api.binance.com 451 from here"),
    Input("OF_SZG", "order_flow", "global_spot", "trade_size_ratio", "z", {"sum": 4, "median": 96},
          False, "none: api.binance.com 451 from here"),
    Input("OF_GUS", "order_flow", "global_spot+us_spot", "global_us_volume", "z", {"sum": 4},
          False, "partial: US leg live via Binance.US REST; global leg none"),
    Input("OF_TBU", "order_flow", "us_spot", "taker_share", "c", {"sum": 4}, False,
          "yes: Binance.US /api/v3/klines"),
    Input("OF_TRU", "order_flow", "us_spot", "trade_count_ratio", "z", {"sum": 4, "median": 96},
          False, "yes: Binance.US /api/v3/klines"),
    Input("PO_OI", "positioning", "um_metrics", "oi_change", "z", {"lag": 16}, True,
          "none on Binance from here; nearest OKX (another venue)"),
    Input("PO_FL", "positioning", "um_metrics", "flush", "z", {"lag": 16}, True,
          "none on Binance from here; nearest OKX (another venue)"),
    Input("PO_TT", "positioning", "um_metrics", "toptrader_change", "z", {"lag": 16}, True,
          "none on Binance from here; nearest OKX (another venue)"),
    Input("PO_TK", "positioning", "um_metrics", "taker_ls", "z", {"snapshots": 12}, True,
          "none on Binance from here; nearest OKX (another venue)"),
    Input("MS_VR", "market_state", "binanceus_ohlcv", "volreg", "z", {"vol": 96, "trail": 2880},
          False, "yes: venue's own candles"),
    Input("MS_BR", "market_state", "binanceus_ohlcv", "breadth", "c", {"ema": 768}, False,
          "yes: venue's own candles"),
    Input("MS_BV", "market_state", "binanceus_ohlcv", "btc_vol", "z", {"vol": 96}, False,
          "yes: venue's own candles"),
    Input("MS_BA", "market_state", "binanceus_ohlcv", "btc_adx", "c", {"adx": 14}, False,
          "yes: venue's own candles"),
    Input("MS_RR", "market_state", "binanceus_ohlcv", "rel_return", "z", {"ret": 96}, False,
          "yes: venue's own candles"),
    Input("BA_LV", "basis", "um_premium", "premium_level", "z", {"mean": 4}, False,
          "none on Binance from here; OKX mark/index nearest"),
    Input("BA_CH", "basis", "um_premium", "premium_change", "z", {"mean": 4, "lag": 16}, False,
          "none on Binance from here; OKX mark/index nearest"),
)
BY_TAG = {i.tag: i for i in TABLE}


def column(tag: str) -> str:
    return f"ni_{tag.lower()}"


def columns(tag: str) -> list[str]:
    """The model/cache columns of one input: the value, plus its missing flag when declared."""
    i = BY_TAG[tag]
    c = column(tag)
    return [c, f"{c}_missing"] if i.missing_flag else [c]


def all_columns() -> list[str]:
    return [c for i in TABLE for c in columns(i.tag)]


def table_sha() -> str:
    blob = json.dumps({"inputs": [asdict(i) for i in TABLE],
                       "consts": [Z_WINDOW, Z_MIN, Z_CLIP, SNAPSHOT_DELAY_MIN, SNAPSHOT_STALE_MIN,
                                  MIN_PAIRS, list(PAIRS), CONTEXT]}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


TABLE_SHA = table_sha()


def symbol(pair: str) -> str:
    """'ZEC/USDT' -> 'ZECUSDT' (archive symbol)."""
    return pair.replace("/", "")


def file_stem(pair: str) -> str:
    """'ZEC/USDT' -> 'ZEC_USDT' (freqtrade data file stem)."""
    return pair.replace("/", "_")
