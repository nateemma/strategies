"""TrainForwardPeakRegressor — standalone trainer for the per-pair
forward-peak-gain regressor used by ``BaseNNMTStrategy``.

For each pair:
  1. Load OHLCV from freqtrade's feather data dir.
  2. Compute indicators via ``DataframePopulator(DatasetType.MINIMAL)``
     so feature columns match what the strategy sees at backtest time.
  3. Fit a RandomForestRegressor on vol-adjusted forward peak gain.
  4. Save to ``user_data/strategies/saved_data/ForwardPeakRegressor/<pair>.joblib``.
  5. Report train R² and Spearman ρ.

Run:
    python user_data/strategies/Debug/TrainForwardPeakRegressor.py \\
        --pairs ICP_USDT ETH_USDT BTC_USDT ADA_USDT LINK_USDT \\
        --timeframe 15m
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

strat_dir = Path(__file__).parent.parent
sys.path.insert(0, str(strat_dir))

from sklearn.metrics import r2_score  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from utils.DataframePopulator import DataframePopulator, DatasetType  # noqa: E402
from utils import ForwardPeakRegressor as FPR  # noqa: E402


def load_pair_data(data_dir: Path, pair: str, timeframe: str) -> pd.DataFrame:
    data_path = data_dir / f"{pair}-{timeframe}.feather"
    if not data_path.exists():
        sys.exit(f"Data file not found: {data_path}")
    df = pd.read_feather(data_path)
    if "date" in df.columns:
        df = df.set_index("date")
    return df


def train_pair(
    df: pd.DataFrame, pair: str, horizon: int, n_estimators: int, max_depth: int,
) -> tuple[float, float, int]:
    """Train + score on the same set. Returns (R², Spearman ρ, n_rows).

    No held-out split here — for the production model we want to use all
    available history. R²/ρ are in-sample diagnostics; use
    ``DebugSignalLearnability --target forward_peak`` for honest CV numbers.
    """
    model = FPR.train(
        df, horizon=horizon, n_estimators=n_estimators, max_depth=max_depth,
    )

    target = FPR.generate_target(df, horizon=horizon)
    target_series = pd.Series(target, index=df.index, name="_target")
    features = FPR.extract_features(df)
    aligned = pd.concat([features, target_series], axis=1).dropna()
    X = aligned.drop(columns=["_target"])
    y = aligned["_target"]
    pred = model.predict(X)

    r2 = float(r2_score(y, pred))
    rho, _ = spearmanr(y, pred)
    rho = float(rho) if not np.isnan(rho) else 0.0
    n_rows = len(aligned)

    storage_root = strat_dir / "saved_data"
    out_path = FPR.model_path(storage_root, pair)
    FPR.save(model, out_path)
    print(f"    saved -> {out_path.relative_to(strat_dir.parent)}")

    return r2, rho, n_rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Train forward-peak-gain regressors per pair.")
    ap.add_argument(
        "--pairs", nargs="+", required=True,
        help="Pairs to train, e.g. ICP_USDT ETH_USDT",
    )
    ap.add_argument(
        "--data-dir",
        default=str(strat_dir.parent / "data" / "binanceus"),
        help="Directory containing <pair>-<timeframe>.feather files",
    )
    ap.add_argument("--timeframe", default="15m")
    ap.add_argument(
        "--horizon", type=int, default=FPR.DEFAULT_HORIZON,
        help=f"Forward window in bars (default {FPR.DEFAULT_HORIZON})",
    )
    ap.add_argument("--n-estimators", type=int, default=200)
    ap.add_argument("--max-depth", type=int, default=10)
    ap.add_argument(
        "--max-bars", type=int, default=0,
        help="If >0, train on only the most recent N bars",
    )
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    pop = DataframePopulator()

    print(f"Training forward-peak regressor: horizon={args.horizon}, "
          f"n_estimators={args.n_estimators}, max_depth={args.max_depth}\n")

    print(f"  {'pair':12s}  {'rows':>8s}  {'R²':>8s}  {'Spearman ρ':>11s}")
    print(f"  {'-'*12}  {'-'*8}  {'-'*8}  {'-'*11}")

    results = []
    for pair in args.pairs:
        df = load_pair_data(data_dir, pair, args.timeframe)
        if args.max_bars > 0 and len(df) > args.max_bars:
            df = df.iloc[-args.max_bars:].copy()
        df = pop.add_indicators(df, dataset_type=DatasetType.MINIMAL)

        r2, rho, n_rows = train_pair(
            df, pair, args.horizon, args.n_estimators, args.max_depth,
        )
        print(f"  {pair:12s}  {n_rows:8d}  {r2:+8.4f}  {rho:+11.4f}")
        results.append((pair, n_rows, r2, rho))

    if results:
        mean_rho = float(np.mean([r[3] for r in results]))
        print(f"\n  mean Spearman ρ across {len(results)} pairs: {mean_rho:+.4f}")


if __name__ == "__main__":
    main()
