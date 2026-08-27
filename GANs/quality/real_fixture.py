"""Real feature matrix for the scorecard (plan Task B1 / risk R1).

Fidelity metrics computed on synthetic-noise fixtures are meaningless: a Gaussian
blob has no heavy tails, no cross-feature structure and no regime shifts, so
every generator looks competent. The scorecard's toy fixture is a CONTRACT
fixture; quality verdicts need this one.

Features come from the production indicator pipeline (`DataframePopulator` +
`FeatureNormalizer.include_list`) over real OHLCV, so the marginals, joints and
tails are the ones the GANs actually face.

LABELS ARE A PROXY. Reproducing the gbb triple-barrier labeller here would couple
the scorecard to strategy internals for no benefit -- the utility probe only needs
labels that are genuinely learnable from the features. Forward-return terciles
satisfy that and are transparent. Read Δval_mcc as "does this synth help predict a
real, learnable target", NOT as a claim about trading edge.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

_STRATS = Path(__file__).resolve().parents[2]
for _p in (str(_STRATS), str(_STRATS / "utils")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DATA_DIR = Path("/Users/philprice95/projects/freqtrade/user_data/data/binanceus")
DEFAULT_PAIRS = ["BTC", "ETH", "SOL", "XRP", "ADA", "LINK"]


def _load_pair(sym: str, timeframe: str = "15m") -> Optional[pd.DataFrame]:
    f = DATA_DIR / f"{sym}_USDT-{timeframe}.feather"
    if not f.exists():
        return None
    d = pd.read_feather(f)
    d["date"] = pd.to_datetime(d["date"], utc=True)
    return d


def build_real_fixture(
    pairs: Optional[List[str]] = None,
    *,
    n_rows: int = 4000,
    horizon: int = 8,
    start: str = "2025-01-01",
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Return (X, one-hot y, feature_names) from the production indicator set."""
    from Framework.FeatureNormalizer import FeatureNormalizer
    # MINIMAL is what BaseNNStrategy actually uses (BaseNNStrategy.py:1266).
    # The DEFAULT path requires a caller-supplied "mid" column and would not
    # match the feature set the GANs are trained on in production.
    from DataframePopulator import DataframePopulator, DatasetType

    wanted = list(FeatureNormalizer.include_list)
    dp = DataframePopulator()
    frames = []
    for sym in (pairs or DEFAULT_PAIRS):
        raw = _load_pair(sym)
        if raw is None or len(raw) < 1000:
            continue
        raw = raw[raw["date"] >= start].reset_index(drop=True)
        if len(raw) < 500:
            continue
        try:
            df = dp.add_indicators(raw.copy(), dataset_type=DatasetType.MINIMAL)
        except Exception:
            continue
        fwd = df["close"].shift(-horizon) / df["close"] - 1
        cols = [c for c in wanted if c in df.columns]
        if len(cols) < 5:
            continue
        sub = df[cols].copy()
        sub["_fwd"] = fwd
        frames.append(sub.replace([np.inf, -np.inf], np.nan).dropna())

    if not frames:
        raise RuntimeError("no usable pairs -- check DATA_DIR and downloaded feathers")

    allf = pd.concat(frames, ignore_index=True)
    feat_cols = [c for c in allf.columns if c != "_fwd"]
    if len(allf) > n_rows:
        allf = allf.sample(n_rows, random_state=seed).reset_index(drop=True)

    x = allf[feat_cols].to_numpy(dtype="float32")
    q = allf["_fwd"].quantile([1 / 3, 2 / 3]).to_numpy()
    cls = np.digitize(allf["_fwd"].to_numpy(), q)      # 0 down / 1 flat / 2 up
    y = np.zeros((len(cls), 3), dtype="float32")
    y[np.arange(len(cls)), cls] = 1.0
    return x, y, feat_cols


if __name__ == "__main__":
    X, Y, names = build_real_fixture()
    print(f"X {X.shape}  y {Y.shape}  classes {Y.sum(0).astype(int)}")
    print(f"features ({len(names)}): {names}")
    print(f"kurtosis (heavy tails?) median={float(pd.DataFrame(X).kurt().median()):.2f}")
