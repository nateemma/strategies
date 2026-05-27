"""
DebugAnalyseIndicators — feature discovery tool.

Different purpose from DebugDfAnalyse: rather than auditing the current
include_list configuration, this tool surveys EVERY numeric column the
DataframePopulator produces to find:
  - High-signal features not yet in include_list
  - Bimodal indicators that would benefit from pos/neg splitting
  - Sparse / heavy-tailed features the GAN will struggle to model
  - Low-signal features currently included that could be dropped

Per-feature metrics:
  Distribution shape           skew, excess kurtosis, bimodality coefficient,
                               sparsity ratio, range/IQR ratio
  Predictive signal            Pearson, Spearman, MI with buy/sell labels
  Modelability score           combined penalty for non-Gaussian shape
  Composite ranking            pred_signal × modelability

Outputs include a dedicated "bimodal candidates" section so split decisions
(like the vwap_ratio → vwap_pos/vwap_neg and macd_norm → macd_pos/macd_neg
that we already applied) can be spotted on new features.

Run:
    python user_data/strategies/Debug/DebugAnalyseIndicators.py \\
        --pair ICP_USDT --timeframe 15m \\
        --save-csv indicator_analysis.csv

This is NOT a freqtrade strategy — it's a standalone analysis tool. Run
it directly with Python from the repo root.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew, spearmanr

# Make project imports work
strat_dir = Path(__file__).parent.parent
sys.path.insert(0, str(strat_dir))

from sklearn.feature_selection import mutual_info_classif  # noqa: E402

from utils.DataframePopulator import DataframePopulator, DatasetType  # noqa: E402
from Framework.TrainingSignals import (  # noqa: E402
    get_train_buy_signals,
    get_train_sell_signals,
    LabelMethod,
    available_methods,
)


# Columns that aren't real model features (raw OHLCV, intermediate
# computations, helper columns). The analysis skips these.
NON_FEATURE_COLS: set = {
    # OHLCV
    "open", "high", "low", "close", "volume",
    # Wavelet / count helpers
    "open_count", "high_count", "low_count", "close_count", "max_count",
    # Intermediate computations
    "close_safe", "log_volume", "gain",
    "bb_lowerband", "bb_middleband", "bb_upperband", "bb_gain", "bb_loss",
    "macd", "macdsignal", "macdhist", "ema_fast",
    "atr_pct", "atr_pct_roll",
    "adx", "aroonosc", "cci", "dymi", "dymi_period", "ss_close",
    "rvol", "sma", "ema",
    # Categorical / binary flags
    "sar_trend", "trend_mode",
    # Custom signal columns (not used as features directly)
    "guard_metric_raw",
}


def load_pair_data(data_dir: Path, pair: str, timeframe: str) -> pd.DataFrame:
    """Load OHLCV from freqtrade's feather format."""
    data_path = data_dir / f"{pair}-{timeframe}.feather"
    if not data_path.exists():
        sys.exit(f"Data file not found: {data_path}")
    df = pd.read_feather(data_path)
    if "date" in df.columns:
        df = df.set_index("date")
    return df


def load_include_list() -> set:
    """Pull the live include_list from BaseNNStrategy so the report can
    mark which features are currently used vs. unused. Best-effort —
    returns an empty set if the import fails."""
    try:
        from Framework.BaseNNStrategy import BaseNNStrategy  # noqa: WPS433
        return set(BaseNNStrategy.include_list)
    except Exception as exc:
        print(f"  (note: couldn't load include_list for comparison: {exc})")
        return set()


def bimodality_coefficient(values: np.ndarray) -> float:
    """Sarle's bimodality coefficient.

    BC = (skew² + 1) / (excess_kurt + 3·(n−1)²/((n−2)(n−3)))

    Interpretation:
        BC > 5/9 (~0.555)   suggests likely bimodal distribution
        BC = 5/9            corresponds to a Gaussian
        Higher values       more bimodal

    Not a hypothesis test — combine with skew/kurt for confidence.
    """
    values = values[~np.isnan(values)]
    n = len(values)
    if n < 4:
        return float("nan")
    s = float(skew(values))
    k = float(kurtosis(values, fisher=True))
    denom = k + 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    if denom == 0:
        return float("nan")
    return (s ** 2 + 1.0) / denom


def sparsity_ratio(values: np.ndarray, band_frac: float = 0.1) -> float:
    """Fraction of values within ``band_frac × std`` of the median.

    High value = the distribution has a spike near the centre relative
    to its spread — the "mostly zero with rare large values" pattern
    that bites the macd_pos / vwap_pos splits. Range [0, 1].
    """
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float("nan")
    med = float(np.median(values))
    std = float(np.std(values))
    if std == 0:
        return 1.0
    return float(np.mean(np.abs(values - med) < band_frac * std))


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation with NaN-handling and constant-input guard."""
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 50:
        return float("nan")
    xx = x[mask]
    yy = y[mask]
    if xx.std() == 0 or yy.std() == 0:
        return 0.0
    return float(np.corrcoef(xx, yy)[0, 1])


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman with NaN handling."""
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 50:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho, _ = spearmanr(x[mask], y[mask])
    return float(rho) if not np.isnan(rho) else 0.0


def safe_mi(x: np.ndarray, y: np.ndarray) -> float:
    """Mutual information with a binary target. NaN-robust."""
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 200:
        return float("nan")
    try:
        return float(
            mutual_info_classif(
                x[mask].reshape(-1, 1),
                y[mask].astype(int),
                random_state=42,
                n_neighbors=3,
            )[0]
        )
    except Exception:
        return float("nan")


def compute_modelability(
    abs_skew: float,
    excess_kurt: float,
    bc: float,
    sparsity: float,
) -> float:
    """Score in [0, 1]. Higher = the GAN can model this distribution
    cleanly. Penalises non-Gaussian shape components.

    Weights chosen empirically:
        skew penalty (×0.30)   — heavy one-sided tails compress badly
        kurt penalty (×0.20)   — fat tails relative to a Gaussian
        bimodality (×0.30)     — the failure mode we already hit on
                                  vwap_ratio / macd_norm
        sparsity (×0.20)       — the macd_pos / vwap_pos overshoot mode
    """
    skew_pen = min(abs_skew, 3.0) / 3.0
    kurt_pen = min(max(excess_kurt, 0.0), 10.0) / 10.0
    bc_pen = max(0.0, bc - 0.555) / (1.0 - 0.555) if not np.isnan(bc) else 0.5
    sparse_pen = sparsity if not np.isnan(sparsity) else 0.5
    score = 1.0 - 0.30 * skew_pen - 0.20 * kurt_pen - 0.30 * bc_pen - 0.20 * sparse_pen
    return float(max(0.0, min(1.0, score)))


def recommend(
    pred_signal: float,
    modelability: float,
    is_bimodal: bool,
    is_sparse: bool,
    already_included: bool,
) -> str:
    """Categorise the feature into an actionable recommendation."""
    if is_bimodal and pred_signal > 0.15:
        return "SPLIT (high signal, bimodal — pos/neg breakdown)"
    if is_sparse and pred_signal > 0.15 and not already_included:
        return "SPLIT or PASSTHROUGH (high signal, sparse)"
    if already_included and pred_signal < 0.05:
        return "DROP (currently in, low signal)"
    if pred_signal > 0.25 and modelability > 0.6 and not already_included:
        return "STRONG ADD (high signal, well-modelable)"
    if pred_signal > 0.15 and not already_included:
        return "ADD CANDIDATE"
    if pred_signal > 0.15 and already_included:
        return "KEEP (in current include_list)"
    if pred_signal < 0.05 and modelability < 0.4:
        return "SKIP (low signal + hard to model)"
    return "MARGINAL"


def analyse_feature(
    name: str,
    values: np.ndarray,
    buy_signals: np.ndarray,
    sell_signals: np.ndarray,
    already_included: bool,
) -> Dict:
    """Compute the full per-feature metric row."""
    # Drop NaNs jointly across feature + both labels.
    mask = ~(np.isnan(values) | np.isnan(buy_signals) | np.isnan(sell_signals))
    n = int(mask.sum())
    if n < 200:
        return {
            "name": name,
            "in_include_list": already_included,
            "n_obs": n,
            "error": "too few non-NaN values",
        }

    v = values[mask]
    b = buy_signals[mask].astype(float)
    s = sell_signals[mask].astype(float)

    # Distribution shape
    sk = float(skew(v))
    kt = float(kurtosis(v, fisher=True))
    bc = bimodality_coefficient(v)
    sp = sparsity_ratio(v)
    q25, q75 = float(np.percentile(v, 25)), float(np.percentile(v, 75))
    iqr = q75 - q25
    v_range = float(v.max() - v.min())
    range_iqr = (v_range / iqr) if iqr > 0 else float("nan")

    # Predictive signal
    buy_corr = safe_corr(v, b)
    sell_corr = safe_corr(v, s)
    buy_sp = safe_spearman(v, b)
    sell_sp = safe_spearman(v, s)
    mi_buy = safe_mi(v, b)
    mi_sell = safe_mi(v, s)

    pred_signal = max(
        abs(buy_corr) if not np.isnan(buy_corr) else 0.0,
        abs(sell_corr) if not np.isnan(sell_corr) else 0.0,
        abs(buy_sp) if not np.isnan(buy_sp) else 0.0,
        abs(sell_sp) if not np.isnan(sell_sp) else 0.0,
    )

    modelability = compute_modelability(abs(sk), kt, bc, sp)
    composite = pred_signal * modelability

    is_bimodal = bool(not np.isnan(bc) and bc > 0.555)
    is_sparse = sp > 0.3
    rec = recommend(pred_signal, modelability, is_bimodal, is_sparse, already_included)

    return {
        "name": name,
        "in_include_list": already_included,
        "n_obs": n,
        "mean": float(np.mean(v)),
        "median": float(np.median(v)),
        "std": float(np.std(v)),
        "skew": sk,
        "excess_kurt": kt,
        "bimodality_coef": bc,
        "is_bimodal": is_bimodal,
        "sparsity": sp,
        "is_sparse": is_sparse,
        "range_iqr_ratio": range_iqr,
        "buy_corr": buy_corr,
        "sell_corr": sell_corr,
        "buy_spearman": buy_sp,
        "sell_spearman": sell_sp,
        "mi_buy": mi_buy,
        "mi_sell": mi_sell,
        "pred_signal": pred_signal,
        "modelability": modelability,
        "composite": composite,
        "recommendation": rec,
    }


def discover_candidate_columns(df: pd.DataFrame) -> List[str]:
    """Identify numeric columns suitable for feature analysis."""
    candidates: List[str] = []
    for col in df.columns:
        if col in NON_FEATURE_COLS:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        # Skip columns that are constant or near-constant
        nunique = df[col].nunique(dropna=True)
        if nunique < 5:
            continue
        candidates.append(col)
    return sorted(candidates)


def fmt_float(v, width: int = 7, places: int = 3) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return " " * (width - 3) + "n/a"
    return f"{v:>{width}.{places}f}"


def print_section(title: str) -> None:
    print()
    print("=" * 90)
    print(title)
    print("=" * 90)


def print_full_table(rows: List[Dict]) -> None:
    """Pretty-print all features ranked by composite score."""
    print_section("FULL FEATURE TABLE (sorted by composite = pred_signal × modelability)")
    valid = [r for r in rows if "error" not in r]
    valid.sort(key=lambda r: r["composite"], reverse=True)

    header = (
        f"  {'feature':22s} {'in?':>4s} {'pred':>6s} {'model':>6s} "
        f"{'compo':>6s} {'skew':>6s} {'kurt':>6s} {'BC':>6s} {'sparse':>6s} "
        f"{'buy_r':>6s} {'sell_r':>6s} {'recommendation'}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in valid:
        in_mark = " yes" if r["in_include_list"] else "  - "
        print(
            f"  {r['name']:22s} {in_mark:>4s} "
            f"{fmt_float(r['pred_signal'], 6, 3)} "
            f"{fmt_float(r['modelability'], 6, 3)} "
            f"{fmt_float(r['composite'], 6, 3)} "
            f"{fmt_float(r['skew'], 6, 2)} "
            f"{fmt_float(r['excess_kurt'], 6, 2)} "
            f"{fmt_float(r['bimodality_coef'], 6, 3)} "
            f"{fmt_float(r['sparsity'], 6, 3)} "
            f"{fmt_float(r['buy_corr'], 6, 3)} "
            f"{fmt_float(r['sell_corr'], 6, 3)} "
            f" {r['recommendation']}"
        )


def print_bimodal_section(rows: List[Dict]) -> None:
    """Specifically surface bimodal candidates — features that would
    benefit from pos/neg splitting (the treatment we already applied to
    vwap_ratio and macd_norm)."""
    print_section("BIMODAL CANDIDATES (BC > 0.555, sorted by predictive signal)")
    bimodal = [r for r in rows if "error" not in r and r.get("is_bimodal")]
    if not bimodal:
        print("  None detected.")
        return
    bimodal.sort(key=lambda r: r["pred_signal"], reverse=True)

    print(f"  {'feature':22s} {'in?':>4s} {'BC':>6s} {'skew':>6s} "
          f"{'sparse':>6s} {'pred':>6s} {'buy_r':>6s} {'sell_r':>6s}  notes")
    print("  " + "-" * 88)
    for r in bimodal:
        note = ""
        if r["pred_signal"] > 0.20:
            note = "high-signal — split likely valuable"
        elif r["pred_signal"] > 0.10:
            note = "moderate signal — consider splitting"
        else:
            note = "low signal — splitting probably not worth it"
        in_mark = " yes" if r["in_include_list"] else "  - "
        print(
            f"  {r['name']:22s} {in_mark:>4s} "
            f"{fmt_float(r['bimodality_coef'], 6, 3)} "
            f"{fmt_float(r['skew'], 6, 2)} "
            f"{fmt_float(r['sparsity'], 6, 3)} "
            f"{fmt_float(r['pred_signal'], 6, 3)} "
            f"{fmt_float(r['buy_corr'], 6, 3)} "
            f"{fmt_float(r['sell_corr'], 6, 3)}  "
            f"{note}"
        )


def print_action_groups(rows: List[Dict]) -> None:
    """Group recommendations into actionable buckets."""
    print_section("ACTIONABLE RECOMMENDATIONS")
    by_rec: Dict[str, List[Dict]] = {}
    for r in rows:
        if "error" in r:
            continue
        by_rec.setdefault(r["recommendation"], []).append(r)

    order = [
        "STRONG ADD (high signal, well-modelable)",
        "ADD CANDIDATE",
        "SPLIT (high signal, bimodal — pos/neg breakdown)",
        "SPLIT or PASSTHROUGH (high signal, sparse)",
        "DROP (currently in, low signal)",
        "SKIP (low signal + hard to model)",
        "KEEP (in current include_list)",
        "MARGINAL",
    ]
    for cat in order:
        items = by_rec.get(cat, [])
        if not items:
            continue
        items.sort(key=lambda r: r["composite"], reverse=True)
        print(f"\n  {cat}:")
        for r in items:
            print(
                f"    {r['name']:22s}  pred={fmt_float(r['pred_signal'], 5, 3)}  "
                f"model={fmt_float(r['modelability'], 5, 3)}  "
                f"BC={fmt_float(r['bimodality_coef'], 5, 3)}  "
                f"sparse={fmt_float(r['sparsity'], 5, 3)}"
            )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Feature discovery: rank ALL DataframePopulator outputs by "
                    "(predictive signal × modelability), with bimodal split flags."
    )
    ap.add_argument("--pair", default="ICP_USDT")
    ap.add_argument(
        "--data-dir",
        default=str(strat_dir.parent / "data" / "binanceus"),
        help="Directory containing <pair>-<timeframe>.feather files",
    )
    ap.add_argument("--timeframe", default="15m")
    ap.add_argument(
        "--label-method",
        type=int,
        default=int(LabelMethod.gbb.value),
        help="LabelMethod ID for buy/sell label generation "
             "(default: 17 = gbb, matches TrainingConfig.TRAINING_TYPE)",
    )
    ap.add_argument(
        "--min-gain",
        type=float,
        default=0.005,
        help="min_gain / min_loss threshold for label generation (default 0.005)",
    )
    ap.add_argument(
        "--horizon",
        type=int,
        default=24,
        help="Forward window (bars) for label generation",
    )
    ap.add_argument(
        "--dataset",
        choices=["MINIMAL", "DEFAULT", "SMALL", "MEDIUM", "LARGE"],
        default="MINIMAL",
        help="DataframePopulator dataset_type — what set of indicators to compute. "
             "MINIMAL matches what production NN strategies use.",
    )
    ap.add_argument(
        "--max-bars", type=int, default=0,
        help="If >0, use only the most recent N bars (speeds up analysis).",
    )
    ap.add_argument("--save-csv", default=None, help="Optional CSV output path.")
    ap.add_argument(
        "--list-methods", action="store_true",
        help="Print available label method IDs and exit.",
    )
    args = ap.parse_args()

    if args.list_methods:
        print("Available label methods:")
        for mid, name in available_methods().items():
            print(f"  {mid:2d}  {name}")
        return

    # Load + populate
    data_dir = Path(args.data_dir)
    print(f"Loading {args.pair} {args.timeframe} from {data_dir}…")
    df = load_pair_data(data_dir, args.pair, args.timeframe)
    if args.max_bars > 0 and len(df) > args.max_bars:
        df = df.iloc[-args.max_bars:].copy()
    print(f"  rows: {len(df)}   {df.index[0]} → {df.index[-1]}")

    print(f"Computing indicators (dataset={args.dataset})…")
    pop = DataframePopulator()
    df = pop.add_indicators(df, dataset_type=getattr(DatasetType, args.dataset))

    # Load include_list for comparison
    include_set = load_include_list()
    print(f"  include_list comparison set: {len(include_set)} features")

    # Identify candidate columns
    candidates = discover_candidate_columns(df)
    print(f"  numeric candidate columns: {len(candidates)} "
          f"({sum(c in include_set for c in candidates)} in include_list, "
          f"{sum(c not in include_set for c in candidates)} not)")

    # Generate labels (gbb / type 17 by default)
    print(f"\nGenerating buy/sell labels via method={args.label_method} "
          f"(min_gain={args.min_gain}, horizon={args.horizon})…")
    label_params = {
        "horizon": args.horizon,
        "min_gain": args.min_gain,
        "min_loss": args.min_gain,
    }
    buy_signals = get_train_buy_signals(
        df, method=args.label_method, params=label_params
    ).astype(float).values
    sell_signals = get_train_sell_signals(
        df, method=args.label_method, params=label_params
    ).astype(float).values
    print(f"  buy signals: {int(buy_signals.sum())} "
          f"({buy_signals.mean() * 100:.1f}%)")
    print(f"  sell signals: {int(sell_signals.sum())} "
          f"({sell_signals.mean() * 100:.1f}%)")

    # Analyse each feature
    print(f"\nAnalysing {len(candidates)} features…")
    rows: List[Dict] = []
    for col in candidates:
        result = analyse_feature(
            col,
            df[col].astype(float).values,
            buy_signals,
            sell_signals,
            already_included=(col in include_set),
        )
        rows.append(result)

    # Drop errored rows from the printed output but keep them in CSV
    errored = [r for r in rows if "error" in r]
    if errored:
        print(f"\n  Skipped {len(errored)} features (too few obs):")
        for r in errored:
            print(f"    {r['name']}: {r['error']}")

    print_full_table(rows)
    print_bimodal_section(rows)
    print_action_groups(rows)

    if args.save_csv:
        out = Path(args.save_csv)
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"\nSaved CSV: {out}")


if __name__ == "__main__":
    main()
