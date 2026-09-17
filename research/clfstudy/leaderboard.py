# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0303, C0411, C0413
# pylint: disable=import-error
# flake8: noqa: F401, E402

"""
Leaderboard: `results.jsonl` -> flat tables + aggregates + plots (T019, data-model.md
§Leaderboard, contracts/clfstudy-cli.md `leaderboard`).

`load_results()` reads the master results file (deduplicating retries, keeping the LAST
line for a given (config_id, split_set, window_id, seed) key). `aggregate()` turns that flat
per-seed table into one row per `config_id`: per-window seed mean/std of every numeric field
(reusing `metrics.summarise_seeds`, so no metric name is hardcoded here) plus pooled figures
(mean over windows of the per-window means/stds), sorted by pooled MCC. `write_outputs()`
renders that into the CSV/Parquet/Markdown/plots this module is responsible for.

This module reads `results.jsonl` only -- it never imports `run.py`/`phase.py` (those are
owned by another concurrent change) and never computes or reports anything about backtest
P&L, trades or drawdown (spec FR-036): every number here is a classification metric against
the fixed gbb labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from research.clfstudy import RESULTS_PATH, REPORT_DIR
from research.clfstudy.metrics import summarise_seeds

PRINCIPLE_III_NOTE = (
    "Classification metrics against fixed gbb labels; NOT evidence of trading edge "
    "(Constitution III)."
)

#: the run-key identity a row's (config_id, split_set, window_id, seed) tuple; a repeated key
#: is a retry -- `load_results` keeps only the last line for it.
KEY_COLS = ("config_id", "split_set", "window_id", "seed")

#: fields whose JSON-decoded value is a list/dict and MUST be kept as a python object (never
#: coerced to numeric) -- data-model.md §RunResult's "matrix"/"probabilistic"/"data" groups.
JSON_COLUMNS = (
    "hp",
    "train",
    "class_counts_train",
    "class_counts_test",
    "pred_counts",
    "confusion",
    "reliability_bins",
    "pairs_included",
    "pairs_excluded",
    "tags",
)

#: scalar numeric RunResult fields; `null` in the JSONL becomes NaN for exactly these.
NUMERIC_COLUMNS = (
    "seed",
    "seq_len",
    "phase",
    "n_train",
    "n_val",
    "n_test",
    "bal_acc",
    "macro_f1",
    "mcc",
    "prec_sell",
    "prec_hold",
    "prec_buy",
    "rec_sell",
    "rec_hold",
    "rec_buy",
    "logloss",
    "auroc_sell",
    "auroc_hold",
    "auroc_buy",
    "auroc_macro",
    "ece",
    "buy_prec_at_08",
    "buy_cov_at_08",
    "n_params",
    "train_seconds",
    "epochs_run",
    "best_epoch",
    "infer_ms_per_window_b256",
    "infer_ms_per_window_b1",
)

#: baseline pseudo/registry archs (research R6, T023): always shown, never a "candidate".
BASELINE_ARCHS = ("MAJORITY", "PERSISTENCE", "RIDGE", "ROCKET")


def load_results(path: str | Path = RESULTS_PATH) -> pd.DataFrame:
    """Read `results.jsonl` into a flat `DataFrame` (data-model.md §RunResult).

    A repeated (config_id, split_set, window_id, seed) key is a retry: only the LAST line for
    that key is kept. JSON `null` becomes `NaN` for the scalar numeric columns
    (`NUMERIC_COLUMNS`); list/dict columns (`JSON_COLUMNS`) are kept as python objects.
    """
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()

    by_key: "dict[tuple, dict]" = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = tuple(row.get(k) for k in KEY_COLS)
            by_key[key] = row  # later line for the same key overwrites -> keeps the LAST

    if not by_key:
        return pd.DataFrame()

    df = pd.DataFrame(list(by_key.values()))
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def rows_for_summary(g: pd.DataFrame) -> list:
    """`g` (one (config_id, window_id) group) as a list of dicts, seed dropped.

    `seed` is the thing being aggregated over, not a metric; every other column is passed to
    `summarise_seeds`, which itself ignores anything non-numeric (status strings, json
    columns, ...) -- so no metric name needs to be hardcoded here.
    """
    cols = [c for c in g.columns if c != "seed"]
    return g[cols].to_dict("records")


def aggregate(df: pd.DataFrame, phase: Optional[int] = None) -> pd.DataFrame:
    """One row per `config_id` (data-model.md §Leaderboard).

    Rows with `status == "failed"` are dropped before aggregation (their metrics are
    meaningless); `collapsed`/`class_absent` rows are kept but flagged via `any_collapsed_<W>`
    / `any_class_absent_<W>` (per window) and `any_collapsed` / `any_class_absent` (pooled,
    True if any window is flagged).

    For every (config_id, window_id) group, every numeric field is aggregated via
    `metrics.summarise_seeds`: `{metric}_mean_<window_id>`, `{metric}_std_<window_id>`
    (ddof=1, NaN with <2 seeds) and `n_seeds_<window_id>`. `pooled_{metric}_mean` is the mean
    over windows of the per-window means; `pooled_{metric}_std` is the mean over windows of
    the per-window stds (both `skipna`, i.e. a window a config has no rows in is simply
    excluded from its pooled figure, judgment call -- documented here since data-model.md only
    says "mean over windows"). `n_windows` counts the windows a config actually has rows in.

    If `phase` is given, only rows with that `phase` are kept EXCEPT the baseline archs
    (`BASELINE_ARCHS`), which are always included regardless of their own echoed phase (they
    are only ever run once, in phase 2, and are shown for reference in every later phase's
    leaderboard).
    """
    if df is None or df.empty or "status" not in df.columns:
        return pd.DataFrame()

    work = df[df["status"] != "failed"].copy()
    if phase is not None:
        work = work[(work["phase"] == phase) | (work["arch"].isin(BASELINE_ARCHS))]
    if work.empty:
        return pd.DataFrame()

    per_window_rows = []
    for (config_id, window_id), g in work.groupby(["config_id", "window_id"], sort=False):
        summary = summarise_seeds(rows_for_summary(g))
        first = g.iloc[0]
        tags = first.get("tags")
        entry = {
            "config_id": config_id,
            "window_id": window_id,
            "arch": first["arch"],
            "hp": json.dumps(first.get("hp") or {}, sort_keys=True, default=str),
            "input_set": first.get("input_set"),
            "seq_len": int(first["seq_len"]) if pd.notna(first.get("seq_len")) else None,
            "phase": int(first["phase"]) if pd.notna(first.get("phase")) else None,
            "tags": json.dumps(list(tags) if isinstance(tags, (list, tuple)) else (tags or [])),
            "any_collapsed": bool((g["status"] == "collapsed").any()),
            "any_class_absent": bool((g["status"] == "class_absent").any()),
        }
        entry.update(summary)
        per_window_rows.append(entry)

    per_window = pd.DataFrame(per_window_rows)
    metric_cols = [c for c in per_window.columns if c.endswith("_mean") or c.endswith("_std")]

    wide_parts = []
    for window_id, g in per_window.groupby("window_id"):
        sub = g.set_index("config_id")[metric_cols + ["n_seeds", "any_collapsed", "any_class_absent"]]
        wide_parts.append(sub.add_suffix(f"_{window_id}"))
    wide = pd.concat(wide_parts, axis=1) if wide_parts else pd.DataFrame()

    echo_cols = ["arch", "hp", "input_set", "seq_len", "phase", "tags"]
    echo = per_window.groupby("config_id")[echo_cols].first()

    pooled: "dict[str, pd.Series]" = {}
    for col in metric_cols:
        pivoted = per_window.pivot(index="config_id", columns="window_id", values=col)
        if col.endswith("_mean"):
            pooled[f"pooled_{col[:-5]}_mean"] = pivoted.mean(axis=1, skipna=True)
        else:
            pooled[f"pooled_{col[:-4]}_std"] = pivoted.mean(axis=1, skipna=True)
    pooled_df = pd.DataFrame(pooled)

    n_windows = per_window.groupby("config_id")["window_id"].nunique().rename("n_windows")
    n_seeds = per_window.groupby("config_id")["n_seeds"].max().rename("n_seeds")
    any_collapsed = per_window.groupby("config_id")["any_collapsed"].any().rename("any_collapsed")
    any_class_absent = (
        per_window.groupby("config_id")["any_class_absent"].any().rename("any_class_absent")
    )

    out = echo.join([wide, pooled_df, n_windows, n_seeds, any_collapsed, any_class_absent])
    out = out.reset_index()
    if "pooled_mcc_mean" in out.columns:
        out = out.sort_values("pooled_mcc_mean", ascending=False, na_position="last")
        out = out.reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# write_outputs: CSV/Parquet + leaderboard markdown + plots
# ---------------------------------------------------------------------------


def _flatten_json_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Copy of `df` with every `JSON_COLUMNS` entry serialised to a JSON string (for CSV/parquet)."""
    flat = df.copy()
    for col in JSON_COLUMNS:
        if col in flat.columns:
            flat[col] = flat[col].map(lambda v: json.dumps(v) if v is not None else None)
    return flat


def _pred_share_table(work: pd.DataFrame) -> pd.DataFrame:
    """Pooled predicted-class share per config: sum(pred_counts) / sum(n_test) (Table 3)."""
    if work.empty or "pred_counts" not in work.columns:
        return pd.DataFrame(columns=["config_id", "sell_share", "hold_share", "buy_share"])
    records = []
    for config_id, g in work.groupby("config_id"):
        counts = np.zeros(3)
        n_test_total = 0.0
        for pc, nt in zip(g["pred_counts"], g["n_test"]):
            if isinstance(pc, (list, tuple)) and len(pc) == 3:
                counts += np.asarray(pc, dtype=float)
            if pd.notna(nt):
                n_test_total += float(nt)
        share = counts / n_test_total if n_test_total > 0 else np.full(3, np.nan)
        records.append(
            {
                "config_id": config_id,
                "sell_share": share[0],
                "hold_share": share[1],
                "buy_share": share[2],
            }
        )
    return pd.DataFrame(records)


def _fmt(x, nd: int = 3) -> str:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "NaN"
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def _fmt_pm(mean, std, nd: int = 3) -> str:
    return f"{_fmt(mean, nd)} ± {_fmt(std, nd)}"


def _render_markdown(agg: pd.DataFrame, work: pd.DataFrame, phase: int) -> str:
    lines = [f"# Leaderboard: phase {phase}", "", PRINCIPLE_III_NOTE, ""]

    if agg.empty:
        lines.append("_No rows for this phase._")
        return "\n".join(lines) + "\n"

    n_seeds_note = ", ".join(
        f"{r.arch}={int(r.n_seeds) if pd.notna(r.n_seeds) else 'NA'}" for r in agg.itertuples()
    )

    window_ids = sorted({c.rsplit("_", 1)[-1] for c in agg.columns if c.startswith("mcc_mean_")})

    # ---- Table 1: pooled leaderboard --------------------------------------------------
    lines += [
        f"## Table 1: pooled leaderboard (seed counts per config: {n_seeds_note})",
        "",
        "| rank | arch | config_id | hp | input_set | seq_len | n_seeds | MCC | macro-F1 |"
        " bal-acc | logloss | ECE | AUROC macro | buy_prec@0.8 | buy_cov@0.8 | n_params |"
        " train (min) | infer ms (b256) | epochs_run | flags |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for rank, row in enumerate(agg.itertuples(), start=1):
        flags = []
        if getattr(row, "any_collapsed", False):
            flags.append("collapsed")
        if getattr(row, "any_class_absent", False):
            flags.append("class_absent")
        if row.arch in BASELINE_ARCHS:
            flags.append("baseline")
        train_min = getattr(row, "pooled_train_seconds_mean", float("nan"))
        train_min = train_min / 60.0 if pd.notna(train_min) else float("nan")
        lines.append(
            "| {rank} | {arch} | {cid} | {hp} | {inp} | {seq} | {ns} | {mcc} | {f1} | {ba} |"
            " {ll} | {ece} | {auroc} | {bp} | {bc} | {npar} | {tmin} | {infer} | {ep} | {fl} |".format(
                rank=rank,
                arch=row.arch,
                cid=str(row.config_id)[:12],
                hp=str(row.hp),
                inp=row.input_set,
                seq=row.seq_len,
                ns=int(row.n_seeds) if pd.notna(row.n_seeds) else "NA",
                mcc=_fmt_pm(getattr(row, "pooled_mcc_mean", float("nan")), getattr(row, "pooled_mcc_std", float("nan"))),
                f1=_fmt_pm(
                    getattr(row, "pooled_macro_f1_mean", float("nan")),
                    getattr(row, "pooled_macro_f1_std", float("nan")),
                ),
                ba=_fmt_pm(
                    getattr(row, "pooled_bal_acc_mean", float("nan")),
                    getattr(row, "pooled_bal_acc_std", float("nan")),
                ),
                ll=_fmt(getattr(row, "pooled_logloss_mean", float("nan"))),
                ece=_fmt(getattr(row, "pooled_ece_mean", float("nan"))),
                auroc=_fmt(getattr(row, "pooled_auroc_macro_mean", float("nan"))),
                bp=_fmt(getattr(row, "pooled_buy_prec_at_08_mean", float("nan"))),
                bc=_fmt(getattr(row, "pooled_buy_cov_at_08_mean", float("nan"))),
                npar=_fmt(getattr(row, "pooled_n_params_mean", float("nan")), 0),
                tmin=_fmt(train_min),
                infer=_fmt(getattr(row, "pooled_infer_ms_per_window_b256_mean", float("nan"))),
                ep=_fmt(getattr(row, "pooled_epochs_run_mean", float("nan")), 1),
                fl=",".join(flags) if flags else "-",
            )
        )

    # ---- Table 2: per-window MCC / macro-F1 --------------------------------------------
    lines += ["", "## Table 2: per-window MCC and macro-F1 (mean ± seed-std)", ""]
    header = "| arch | config_id | " + " | ".join(f"MCC {w}" for w in window_ids) + " | "
    header += " | ".join(f"macro-F1 {w}" for w in window_ids) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (2 + 2 * len(window_ids)))
    for row in agg.itertuples():
        cells = [row.arch, str(row.config_id)[:12]]
        for w in window_ids:
            cells.append(
                _fmt_pm(
                    getattr(row, f"mcc_mean_{w}", float("nan")),
                    getattr(row, f"mcc_std_{w}", float("nan")),
                )
            )
        for w in window_ids:
            cells.append(
                _fmt_pm(
                    getattr(row, f"macro_f1_mean_{w}", float("nan")),
                    getattr(row, f"macro_f1_std_{w}", float("nan")),
                )
            )
        lines.append("| " + " | ".join(str(c) for c in cells) + " |")

    # ---- Table 3: predicted-class share -------------------------------------------------
    share = _pred_share_table(work)
    lines += ["", "## Table 3: predicted-class share (pooled pred_counts / n_test)", ""]
    lines.append("| config_id | arch | SELL share | HOLD share | BUY share |")
    lines.append("|---|---|---|---|---|")
    arch_by_cfg = dict(zip(agg["config_id"], agg["arch"]))
    for r in share.itertuples():
        lines.append(
            "| {cid} | {arch} | {s} | {h} | {b} |".format(
                cid=str(r.config_id)[:12],
                arch=arch_by_cfg.get(r.config_id, "?"),
                s=_fmt(r.sell_share),
                h=_fmt(r.hold_share),
                b=_fmt(r.buy_share),
            )
        )

    return "\n".join(lines) + "\n"


def _plot_mcc_bar(agg: pd.DataFrame, phase: int, out_path: Path) -> None:
    if agg.empty or "pooled_mcc_mean" not in agg.columns:
        return
    sub = agg.dropna(subset=["pooled_mcc_mean"])
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(max(6, len(sub) * 0.6), 5))
    colors = ["tab:orange" if a in BASELINE_ARCHS else "tab:blue" for a in sub["arch"]]
    yerr = sub["pooled_mcc_std"].fillna(0.0) if "pooled_mcc_std" in sub.columns else None
    ax.bar(range(len(sub)), sub["pooled_mcc_mean"].to_numpy(), yerr=yerr, color=colors)
    ax.set_xticks(range(len(sub)))
    ax.set_xticklabels(
        [f"{a}\n{c[:6]}" for a, c in zip(sub["arch"], sub["config_id"].astype(str))],
        rotation=90,
        fontsize=7,
    )
    ax.set_ylabel("pooled MCC (mean ± seed-std)")
    ax.set_title(f"Phase {phase}: pooled MCC — {PRINCIPLE_III_NOTE}", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_mcc_vs_params(agg: pd.DataFrame, phase: int, out_path: Path) -> None:
    if agg.empty or "pooled_n_params_mean" not in agg.columns or "pooled_mcc_mean" not in agg.columns:
        return
    sub = agg.dropna(subset=["pooled_n_params_mean", "pooled_mcc_mean"])
    sub = sub[sub["pooled_n_params_mean"] > 0]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(sub["pooled_n_params_mean"], sub["pooled_mcc_mean"])
    for _, row in sub.iterrows():
        ax.annotate(row["arch"], (row["pooled_n_params_mean"], row["pooled_mcc_mean"]), fontsize=6)
    ax.set_xscale("log")
    ax.set_xlabel("n_params (log scale)")
    ax.set_ylabel("pooled MCC")
    ax.set_title(f"Phase {phase}: pooled MCC vs model size")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_mcc_heatmap(agg: pd.DataFrame, phase: int, out_path: Path) -> None:
    window_ids = sorted({c.rsplit("_", 1)[-1] for c in agg.columns if c.startswith("mcc_mean_")})
    if agg.empty or not window_ids:
        return
    cols = [f"mcc_mean_{w}" for w in window_ids]
    matrix = agg[cols].to_numpy(dtype=float)
    if not np.isfinite(matrix).any():
        return
    fig, ax = plt.subplots(figsize=(max(4, len(window_ids) * 1.5), max(4, len(agg) * 0.4)))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(window_ids)))
    ax.set_xticklabels(window_ids)
    ax.set_yticks(range(len(agg)))
    ax.set_yticklabels([f"{a}:{c[:6]}" for a, c in zip(agg["arch"], agg["config_id"].astype(str))], fontsize=6)
    ax.set_title(f"Phase {phase}: MCC by config x window")
    fig.colorbar(im, ax=ax, label="MCC (seed-mean)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_reliability(df: pd.DataFrame, agg: pd.DataFrame, phase: int, out_path: Path) -> None:
    if agg.empty or "pooled_mcc_mean" not in agg.columns or "reliability_bins" not in df.columns:
        return
    top = agg.dropna(subset=["pooled_mcc_mean"]).sort_values("pooled_mcc_mean", ascending=False).head(5)
    if top.empty:
        return
    fig, axes = plt.subplots(1, len(top), figsize=(4 * len(top), 4), squeeze=False)
    for ax, row in zip(axes[0], top.itertuples()):
        cfg = row.config_id
        rows = df[
            (df["config_id"] == cfg)
            & (df.get("seed", pd.Series(dtype=float)) == 0)
            & (df["status"] != "failed")
        ]
        bins_by_window = [b for b in rows["reliability_bins"] if isinstance(b, list) and b]
        if not bins_by_window:
            ax.set_visible(False)
            continue
        n_bins = len(bins_by_window[0])
        conf = np.full((len(bins_by_window), n_bins), np.nan)
        acc = np.full((len(bins_by_window), n_bins), np.nan)
        for i, bins in enumerate(bins_by_window):
            for j, b in enumerate(bins):
                conf[i, j] = b.get("conf", np.nan)
                acc[i, j] = b.get("acc", np.nan)
        with np.errstate(invalid="ignore"):
            conf_mean = np.nanmean(conf, axis=0)
            acc_mean = np.nanmean(acc, axis=0)
        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        ax.plot(conf_mean, acc_mean, "o-")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"{row.arch}\n{str(cfg)[:8]}", fontsize=8)
        ax.set_xlabel("confidence")
        ax.set_ylabel("accuracy")
    fig.suptitle(f"Phase {phase}: reliability (top {len(top)} by pooled MCC)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def write_outputs(
    phase: int,
    out_dir: str | Path = REPORT_DIR,
    results_path: str | Path = RESULTS_PATH,
) -> dict:
    """Write `results.{csv,parquet}`, `leaderboard_<phase>.{csv,md}` and the plots.

    `results.{csv,parquet}` are the FULL flat table (every phase, every status) written next
    to `results_path`; `leaderboard_<phase>.*` and the plots are phase-filtered (baselines
    always included) and written under `out_dir`. Every plot function checks its own
    preconditions and returns (writing nothing) rather than raising when the phase's data
    can't support it (e.g. an all-NaN column) -- "skip gracefully" per the task.

    `results_path` defaults to the module's `RESULTS_PATH` but is a real parameter (not just
    read off the module global at call time) so tests can point it at a tmp file.

    Returns a dict of the paths written (values are `None` for any plot that was skipped).
    """
    out_dir = Path(out_dir)
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(results_path)

    results_dir = Path(results_path).parent
    results_dir.mkdir(parents=True, exist_ok=True)
    results_csv = results_dir / "results.csv"
    results_parquet = results_dir / "results.parquet"
    if not df.empty:
        flat = _flatten_json_columns(df)
        flat.to_csv(results_csv, index=False)
        try:
            flat.to_parquet(results_parquet, index=False)
        except Exception as exc:  # pragma: no cover - defensive, e.g. missing pyarrow
            print(f"leaderboard: skipping results.parquet ({exc})")
            results_parquet = None
    else:
        results_csv = None
        results_parquet = None

    work = df.copy()
    if not work.empty:
        work = work[work["status"] != "failed"]
        if phase is not None:
            work = work[(work["phase"] == phase) | (work["arch"].isin(BASELINE_ARCHS))]

    agg = aggregate(df, phase=phase)

    leaderboard_csv = out_dir / f"leaderboard_{phase}.csv"
    agg.to_csv(leaderboard_csv, index=False)

    leaderboard_md = out_dir / f"leaderboard_{phase}.md"
    leaderboard_md.write_text(_render_markdown(agg, work, phase))

    plot_paths = {
        "mcc_bar": plots_dir / f"mcc_bar_{phase}.png",
        "mcc_vs_params": plots_dir / f"mcc_vs_params_{phase}.png",
        "mcc_heatmap": plots_dir / f"mcc_heatmap_{phase}.png",
        "reliability": plots_dir / f"reliability_{phase}.png",
    }
    plot_fns = {
        "mcc_bar": lambda p: _plot_mcc_bar(agg, phase, p),
        "mcc_vs_params": lambda p: _plot_mcc_vs_params(agg, phase, p),
        "mcc_heatmap": lambda p: _plot_mcc_heatmap(agg, phase, p),
        "reliability": lambda p: _plot_reliability(work, agg, phase, p),
    }
    written = {}
    for key, path in plot_paths.items():
        try:
            plot_fns[key](path)
            written[key] = str(path) if path.exists() else None
        except Exception as exc:  # pragma: no cover - defensive "skip gracefully"
            print(f"leaderboard: skipping plot {key} ({exc})")
            written[key] = None

    return {
        "results_csv": str(results_csv) if results_csv else None,
        "results_parquet": str(results_parquet) if results_parquet else None,
        "leaderboard_csv": str(leaderboard_csv),
        "leaderboard_md": str(leaderboard_md),
        "plots": written,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="research.clfstudy leaderboard")
    parser.add_argument("--phase", type=int, default=None)
    parser.add_argument("--out", type=str, default=str(REPORT_DIR))
    args = parser.parse_args()
    paths = write_outputs(args.phase, out_dir=args.out)
    for k, v in paths.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
