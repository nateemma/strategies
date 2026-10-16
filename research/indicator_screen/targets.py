"""037 targets (research R4, R11): the gbb_path labels and the regime targets A1/A2/B/C.

Labels (`--labels`): the ARO control class's own `get_training_labels` on each pair's full-range
frame (the classes the ARO model trains on), stored into the harvest cache as `label:buy` /
`label:sell` (NaN for the last 48 bars, which have no future).

Regime targets are built on demand by `regime_target` (they depend on the fold's cut points).

    PYTHONPATH=. ../../.venv/bin/python -m research.indicator_screen.targets --labels
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from research.indicator_screen import harvest as H
from research.indicator_screen import windows as W


# ---------------------------------------------------------------------------- labels


def labels_for(aro, ohlcv: pd.DataFrame, pair: str) -> tuple[np.ndarray, np.ndarray]:
    """(buy, sell) float arrays in {0, 1, NaN}: the ARO control's trained classes."""
    from Framework.BaseStrategy import TradingAction

    d = H.populate_path(aro, ohlcv, pair)
    k = np.asarray(aro.get_training_labels(d.copy()))[-len(d):]
    buy = (k == int(TradingAction.BUY)).astype(float)
    sell = (k == int(TradingAction.SELL)).astype(float)
    buy[-W.LABEL_HORIZON:] = np.nan
    sell[-W.LABEL_HORIZON:] = np.nan
    return buy, sell


def add_labels(pairs: list[str]) -> None:
    H._quiet()
    cfg = H.base_config()
    aro = H.load_aro()
    for pair in pairs:
        frame = H.read_cache(pair)
        buy, sell = labels_for(aro, H.load_ohlcv(cfg, pair), pair)
        if len(buy) != len(frame):
            raise SystemExit(f"{pair}: label rows {len(buy)} != cache rows {len(frame)}")
        frame["label:buy"] = buy.astype(np.float32)
        frame["label:sell"] = sell.astype(np.float32)
        sc = json.loads(H.sidecar_path(pair).read_text())
        frame.to_feather(H.cache_path(pair))
        sc["columns"] = [c for c in frame.columns if c not in ("date", *H.OHLCV)]
        sc["labels"] = {"class": H.ARO_CLASS, "buy_share": float(np.nanmean(buy)),
                        "sell_share": float(np.nanmean(sell))}
        H.sidecar_path(pair).write_text(json.dumps(sc, indent=1))
        print(f"{pair}: buy {np.nanmean(buy):.4f} sell {np.nanmean(sell):.4f}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--labels", action="store_true", required=True)
    ap.add_argument("--pair", default="all")
    a = ap.parse_args(argv)
    cfg = H.base_config()
    pl = [p for p in H.pairs(cfg) if p != H.BTC] if a.pair == "all" else [a.pair]
    add_labels(pl)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
