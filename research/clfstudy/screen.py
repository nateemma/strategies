# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0303, C0411, C0413
# pylint: disable=import-error
# flake8: noqa: F401, E402

"""
Screen: the author's label/input-variant screening rule (feature 019 follow-ups).

A "variant" is a distinct `label` dict (and/or `input_set` / `seq_len`) run, within one phase
(20+), as RIDGE (3 seeds) and MLP_FAM width 64 (2 seeds) over windows A/B/C -- the cheap
2-arch screen `gates/review_19.md` s.6 proposed instead of the full six-type grid. `screen()`
reads `results.jsonl` (via `leaderboard.load_results`, never `run.py`/`phase.py`/`registry.py`)
and applies the author's promotion rule verbatim:

    PROMOTE if MLP - RIDGE >= 0.02 MCC on >= 2 of 3 windows, OR |ridge_shift| >= 0.05
    (ridge_shift = variant RIDGE pooled MCC - baseline-phase RIDGE pooled MCC); else NO.

A variant missing RIDGE or MLP_FAM rows entirely is INCOMPLETE (not scored either way).

Purely a classification-metric screen (data-model.md §RunResult's mcc field): nothing here
computes or reports backtest P&L, trades or drawdown (spec FR-036, Constitution III).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from research.clfstudy import REPORT_DIR, RESULTS_PATH
from research.clfstudy.leaderboard import load_results

RIDGE_ARCH = "RIDGE"
MLP_ARCH = "MLP_FAM"

#: the author's promotion thresholds (gates/review_19.md s.6), verbatim
MCC_DELTA_THRESHOLD = 0.02
MIN_WINDOWS_FOR_DELTA = 2
RIDGE_SHIFT_THRESHOLD = 0.05

DEFAULT_BASELINE_PHASE = 19


def _canonical_label(label: Any) -> str:
    """Stable grouping key for a row's `label` field: canonical (sort_keys) JSON.

    A missing/non-dict `label` (older rows / rows that predate the field) normalises to the
    `{"name": "gbb"}` default, same convention as `leaderboard._label_ident`.
    """
    label = label if isinstance(label, dict) and label else {"name": "gbb"}
    return json.dumps(label, sort_keys=True, default=str)


def _label_display(label: Any) -> str:
    """Short display string for a row's `label` field.

    Delegates to `labels.label_key` when importable (lazy import: another agent owns
    labels.py) so the report reads e.g. `tb_h48_u0.02_d0.02_cr1_pf0`; falls back to the
    canonical JSON if the import or the key computation fails.
    """
    label = label if isinstance(label, dict) and label else None
    try:
        from research.clfstudy import labels as labels_mod

        return labels_mod.label_key(label)
    except Exception:
        return _canonical_label(label)


def _seed_mean_by_window(rows: pd.DataFrame) -> Dict[str, float]:
    """`{window_id: seed-mean mcc}` for one arch's rows within a variant."""
    if rows.empty:
        return {}
    return {str(w): float(v) for w, v in rows.groupby("window_id")["mcc"].mean().items()}


def _pooled(mean_by_window: Dict[str, float]) -> float:
    """Mean over windows of the per-window seed-means; NaN if there are none."""
    if not mean_by_window:
        return float("nan")
    return float(np.mean(list(mean_by_window.values())))


def _class_share(row: pd.Series) -> Optional[np.ndarray]:
    """`class_counts_test` normalised to shares, or `None` if absent/degenerate."""
    counts = row.get("class_counts_test")
    if not isinstance(counts, (list, tuple)) or len(counts) != 3:
        return None
    counts = np.asarray(counts, dtype=float)
    total = counts.sum()
    if total <= 0:
        return None
    return counts / total


def _variant_balance(g: pd.DataFrame) -> Optional[List[float]]:
    """Mean SELL/HOLD/BUY test-class share across every non-failed row in the variant.

    `class_counts_test` depends on the label and window, not the arch, so RIDGE and MLP_FAM
    rows for the same window/label are averaged together (they describe the same test data).
    """
    shares = [s for s in (_class_share(row) for _, row in g[g["status"] != "failed"].iterrows()) if s is not None]
    if not shares:
        return None
    return [float(x) for x in np.mean(shares, axis=0)]


def _baseline_ridge_pooled(df: pd.DataFrame, baseline_phase: int) -> float:
    """Pooled (mean-over-windows) seed-mean RIDGE MCC from `baseline_phase`'s own RIDGE rows."""
    if df.empty or "arch" not in df.columns:
        return float("nan")
    base = df[(df["arch"] == RIDGE_ARCH) & (df["phase"] == baseline_phase) & (df["status"] != "failed")]
    return _pooled(_seed_mean_by_window(base))


def _summarise_variant(g: pd.DataFrame, baseline_pooled: float) -> Dict[str, Any]:
    """One variant's group of rows (already restricted to RIDGE/MLP_FAM, one phase) -> report row."""
    first = g.iloc[0]
    ridge_rows = g[(g["arch"] == RIDGE_ARCH) & (g["status"] != "failed")]
    mlp_rows = g[(g["arch"] == MLP_ARCH) & (g["status"] != "failed")]

    ridge_means = _seed_mean_by_window(ridge_rows)
    mlp_means = _seed_mean_by_window(mlp_rows)
    ridge_pooled = _pooled(ridge_means)
    mlp_pooled = _pooled(mlp_means)

    windows = sorted(set(ridge_means) | set(mlp_means))
    deltas: Dict[str, Optional[float]] = {}
    pass_count = 0
    for w in windows:
        if w in ridge_means and w in mlp_means:
            d = mlp_means[w] - ridge_means[w]
            deltas[w] = d
            if d >= MCC_DELTA_THRESHOLD:
                pass_count += 1
        else:
            deltas[w] = None

    ridge_shift = ridge_pooled - baseline_pooled if np.isfinite(ridge_pooled) and np.isfinite(baseline_pooled) else float("nan")

    any_failed = bool((g["status"] == "failed").any())
    any_collapsed = bool((g["status"] == "collapsed").any())
    notes: List[str] = []
    if any_failed:
        notes.append("failed rows present")
    if any_collapsed:
        notes.append("collapsed rows present")

    has_ridge = not ridge_rows.empty
    has_mlp = not mlp_rows.empty

    if not has_ridge or not has_mlp:
        verdict = "INCOMPLETE"
        missing = [name for name, present in ((RIDGE_ARCH, has_ridge), (MLP_ARCH, has_mlp)) if not present]
        notes.append(f"missing {' and '.join(missing)} rows")
    else:
        promote_by_delta = pass_count >= MIN_WINDOWS_FOR_DELTA
        promote_by_shift = bool(np.isfinite(ridge_shift) and abs(ridge_shift) >= RIDGE_SHIFT_THRESHOLD)
        if promote_by_shift and ridge_shift < 0:
            notes.append("label harder: large negative ridge_shift")
        verdict = "PROMOTE" if (promote_by_delta or promote_by_shift) else "NO"

    label = first.get("label")
    return {
        "label": label if isinstance(label, dict) and label else {"name": "gbb"},
        "variant_display": _label_display(label),
        "input_set": first.get("input_set"),
        "seq_len": int(first["seq_len"]) if pd.notna(first.get("seq_len")) else None,
        "ridge": {"by_window": ridge_means, "pooled": ridge_pooled},
        "mlp": {"by_window": mlp_means, "pooled": mlp_pooled},
        "delta": deltas,
        "windows_passing_delta": pass_count,
        "ridge_shift": ridge_shift,
        "balance": _variant_balance(g),
        "verdict": verdict,
        "notes": notes,
    }


def screen(df: pd.DataFrame, phase: int, baseline_phase: int = DEFAULT_BASELINE_PHASE) -> Dict[str, Any]:
    """Apply the promotion rule to every (label, input_set, seq_len) variant in `phase`.

    Returns a dict with `phase`, `baseline_phase`, `baseline_ridge_pooled` and `variants`
    (a list of per-variant dicts, sorted PROMOTE-first then by `variant_display`).
    """
    baseline_pooled = _baseline_ridge_pooled(df, baseline_phase)
    result: Dict[str, Any] = {
        "phase": phase,
        "baseline_phase": baseline_phase,
        "baseline_ridge_pooled": baseline_pooled,
        "variants": [],
    }
    if df.empty or "phase" not in df.columns:
        return result

    phase_rows = df[(df["phase"] == phase) & (df["arch"].isin((RIDGE_ARCH, MLP_ARCH)))].copy()
    if phase_rows.empty:
        return result

    label_col = phase_rows["label"] if "label" in phase_rows.columns else pd.Series([None] * len(phase_rows), index=phase_rows.index)
    phase_rows["_label_canon"] = label_col.map(_canonical_label)

    variants = [
        _summarise_variant(g, baseline_pooled)
        for _, g in phase_rows.groupby(["_label_canon", "input_set", "seq_len"], sort=False, dropna=False)
    ]
    variants.sort(key=lambda v: (v["verdict"] != "PROMOTE", v["variant_display"], v["input_set"], v["seq_len"]))
    result["variants"] = variants
    return result


# ---------------------------------------------------------------------------
# rendering + CLI
# ---------------------------------------------------------------------------


def _fmt(x, nd: int = 3) -> str:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "NaN"
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def _fmt_by_window(by_window: Dict[str, float], window_ids: List[str]) -> List[str]:
    return [_fmt(by_window.get(w)) for w in window_ids]


def render_markdown(result: Dict[str, Any]) -> str:
    phase = result["phase"]
    baseline_phase = result["baseline_phase"]
    baseline_pooled = result["baseline_ridge_pooled"]
    variants = result["variants"]

    lines = [
        f"# Screen: phase {phase}",
        "",
        f"Label/input-variant screen (RIDGE + MLP_FAM w64, {RIDGE_ARCH}/{MLP_ARCH} only) against "
        f"baseline phase {baseline_phase}'s RIDGE pooled MCC = {_fmt(baseline_pooled)} "
        "(gates/review_19.md s.6 rule). Classification metrics only; NOT evidence of trading "
        "edge (Constitution III).",
        "",
        f"Rule: PROMOTE if MLP - RIDGE >= {MCC_DELTA_THRESHOLD} on >= {MIN_WINDOWS_FOR_DELTA} of 3 "
        f"windows, OR |ridge_shift| >= {RIDGE_SHIFT_THRESHOLD}; else NO. A variant missing RIDGE "
        "or MLP_FAM rows entirely is INCOMPLETE.",
        "",
    ]

    if not variants:
        lines.append("_No RIDGE/MLP_FAM rows for this phase._")
        return "\n".join(lines) + "\n"

    window_ids = sorted({w for v in variants for w in list(v["ridge"]["by_window"]) + list(v["mlp"]["by_window"])})

    header = (
        "| variant | input_set | seq_len | "
        + " | ".join(f"RIDGE {w}" for w in window_ids)
        + " | RIDGE pooled | "
        + " | ".join(f"MLP {w}" for w in window_ids)
        + " | MLP pooled | "
        + " | ".join(f"delta {w}" for w in window_ids)
        + " | ridge_shift | balance (S/H/B) | verdict | notes |"
    )
    lines.append(header)
    lines.append("|" + "---|" * (3 + 3 * len(window_ids) + 5))

    for v in variants:
        balance = v["balance"]
        balance_str = "/".join(_fmt(x, 2) for x in balance) if balance else "NaN"
        cells = [
            v["variant_display"],
            str(v["input_set"]),
            str(v["seq_len"]),
            *_fmt_by_window(v["ridge"]["by_window"], window_ids),
            _fmt(v["ridge"]["pooled"]),
            *_fmt_by_window(v["mlp"]["by_window"], window_ids),
            _fmt(v["mlp"]["pooled"]),
            *[_fmt(v["delta"].get(w)) for w in window_ids],
            _fmt(v["ridge_shift"]),
            balance_str,
            v["verdict"],
            "; ".join(v["notes"]) if v["notes"] else "-",
        ]
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines) + "\n"


def write_screen(
    phase: int,
    baseline_phase: int = DEFAULT_BASELINE_PHASE,
    results_path: "str | Path" = RESULTS_PATH,
    out_dir: "str | Path" = REPORT_DIR,
) -> Dict[str, Any]:
    """`screen()` + write `report/screen_<phase>.{md,json}`; returns the result dict."""
    df = load_results(results_path)
    result = screen(df, phase, baseline_phase=baseline_phase)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"screen_{phase}.md").write_text(render_markdown(result))
    (out_dir / f"screen_{phase}.json").write_text(json.dumps(result, indent=2, sort_keys=False))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="research.clfstudy label/input variant screen")
    parser.add_argument("--phase", type=int, required=True)
    parser.add_argument("--baseline-phase", type=int, default=DEFAULT_BASELINE_PHASE)
    parser.add_argument("--results-path", type=str, default=str(RESULTS_PATH))
    parser.add_argument("--out", type=str, default=str(REPORT_DIR))
    args = parser.parse_args()

    result = write_screen(
        args.phase,
        baseline_phase=args.baseline_phase,
        results_path=args.results_path,
        out_dir=args.out,
    )
    print(render_markdown(result))

    promoted = [v["variant_display"] for v in result["variants"] if v["verdict"] == "PROMOTE"]
    print(f"PROMOTE ({len(promoted)}): {', '.join(promoted) if promoted else 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
