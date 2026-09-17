# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0303, C0411, C0413
# pylint: disable=import-error
# flake8: noqa: F401, E402

"""
Gate: FR-028 viability decisions (T020, data-model.md §GateDecision, contracts/schemas.md
`GateDecision`, contracts/clfstudy-cli.md `gate`).

`decide()` reads `results.jsonl` (via `leaderboard.load_results`, never `run.py`/`phase.py`)
and applies the FR-028 rule: a config is viable when, on a MAJORITY of evaluation windows,
its seed-mean MCC and seed-mean macro-F1 both beat the RIDGE baseline by at least the
adaptive margin, its across-seed MCC std is within the stability bound on EVERY window, and
it carries no `collapsed`/`failed` row on ANY window. `write_decision()` writes
`gates/phase_N.{json,md}` and prints the proposed viable set.

`MAJORITY`, `PERSISTENCE`, `RIDGE` and `ROCKET` are shown for reference but are never
"viable" -- they are baselines, not candidates (data-model.md §GateDecision).
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from research.clfstudy import GATES_DIR, RESULTS_PATH
from research.clfstudy.leaderboard import BASELINE_ARCHS, rows_for_summary, load_results
from research.clfstudy.metrics import summarise_seeds


def _window_clause(g: pd.DataFrame, ridge_ref: dict, margin: float, std_bound: float) -> dict:
    """FR-028's four clauses for one (config, window) group `g` (raw rows, ALL statuses)."""
    status_ok = not g["status"].isin(["collapsed", "failed"]).any()
    non_failed = g[g["status"] != "failed"]

    if non_failed.empty:
        return {
            "mcc_ok": False,
            "f1_ok": False,
            "std_ok": False,
            "status_ok": bool(status_ok),
            "mcc_mean": float("nan"),
            "macro_f1_mean": float("nan"),
            "std_mcc": float("nan"),
            "n_seeds": 0,
            "window_pass": False,
            "notes": ["no_valid_seeds"],
        }

    summary = summarise_seeds(rows_for_summary(non_failed))
    mcc_mean = summary.get("mcc_mean", float("nan"))
    f1_mean = summary.get("macro_f1_mean", float("nan"))
    std_mcc = summary.get("mcc_std", float("nan"))
    n_seeds = int(summary.get("n_seeds", 0))

    notes: list = []
    mcc_ok = bool(
        pd.notna(mcc_mean) and pd.notna(ridge_ref["mcc"]) and mcc_mean >= ridge_ref["mcc"] + margin
    )
    f1_ok = bool(
        pd.notna(f1_mean)
        and pd.notna(ridge_ref["macro_f1"])
        and f1_mean >= ridge_ref["macro_f1"] + margin
    )
    if n_seeds < 2 or pd.isna(std_mcc):
        std_ok = False
        notes.append("insufficient_seeds")
    else:
        std_ok = bool(std_mcc <= std_bound)

    window_pass = bool(mcc_ok and f1_ok and std_ok and status_ok)
    return {
        "mcc_ok": mcc_ok,
        "f1_ok": f1_ok,
        "std_ok": std_ok,
        "status_ok": bool(status_ok),
        "mcc_mean": mcc_mean,
        "macro_f1_mean": f1_mean,
        "std_mcc": std_mcc,
        "n_seeds": n_seeds,
        "window_pass": window_pass,
        "notes": notes,
    }


def decide(
    phase: int,
    margin_floor: float = 0.02,
    std_bound: float = 0.03,
    majority_of: Optional[int] = None,
    results_path: "str | Path" = RESULTS_PATH,
    std_mode: str = "every",
) -> dict:
    """Compute the FR-028 `GateDecision` for `phase` (does not write anything).

    `margin_used = max(margin_floor, 2 * ridge_pooled_seed_std_mcc)`, where
    `ridge_pooled_seed_std_mcc` is the mean (skipping windows with <2 seeds, hence an
    undefined std) over windows of RIDGE's per-window across-seed MCC std (ddof=1). The
    RIDGE reference is taken from every `arch == "RIDGE"` row regardless of its own echoed
    `phase` (RIDGE is only ever run once, in phase 2, per research R6/T023).

    `majority_of` defaults to the number of windows RIDGE itself has rows in (normally 3,
    A/B/C) -- documented judgment call, since data-model.md doesn't pin this down explicitly.

    `std_mode` selects how the stability clause gates viability (FR-028(ii)):
    - `"every"` (default, the spec rule): `viable` requires `std_ok` on EVERY window, checked
      independently of the margin-majority clause. A config that clears the margin on a
      majority of windows but has one unstable window (`std_ok` False) is NOT viable; it is
      reported instead in `stability_only_failures` as a third-seed candidate (FR-004).
    - `"majority"`: reproduces the pre-FR-028-fix behaviour, folding `std_ok` into the
      per-window `window_pass` clause so instability on one window can be outvoted by a
      majority of stable, passing windows.
    """
    if std_mode not in ("every", "majority"):
        raise ValueError(f"std_mode must be 'every' or 'majority', got {std_mode!r}")
    df = load_results(results_path)
    if df.empty:
        raise ValueError(f"no results found at {results_path}")

    ridge_df = df[(df["arch"] == "RIDGE") & (df["status"] != "failed")]
    if ridge_df.empty:
        raise ValueError("no RIDGE baseline rows found in results; run phase 2 first")

    ridge_windows: "dict[str, dict]" = {}
    for window_id, g in ridge_df.groupby("window_id"):
        summary = summarise_seeds(rows_for_summary(g))
        ridge_windows[window_id] = {
            "mcc": summary.get("mcc_mean", float("nan")),
            "macro_f1": summary.get("macro_f1_mean", float("nan")),
            "std_mcc": summary.get("mcc_std", float("nan")),
            "n_seeds": int(summary.get("n_seeds", 0)),
        }

    std_values = [v["std_mcc"] for v in ridge_windows.values() if pd.notna(v["std_mcc"])]
    ridge_pooled_seed_std_mcc = float(np.mean(std_values)) if std_values else float("nan")
    margin_used = (
        max(margin_floor, 2.0 * ridge_pooled_seed_std_mcc)
        if pd.notna(ridge_pooled_seed_std_mcc)
        else margin_floor
    )

    window_ids = sorted(ridge_windows.keys())
    if majority_of is None:
        majority_of = len(window_ids)

    candidate_ids = set(df.loc[df["phase"] == phase, "config_id"].dropna().unique())
    baseline_ids = set(df.loc[df["arch"].isin(BASELINE_ARCHS), "config_id"].dropna().unique())
    all_ids = candidate_ids | baseline_ids

    per_config: "dict[str, dict]" = {}
    stability_only_failures: "list[dict]" = []
    for cfg in sorted(all_ids):
        cfg_rows = df[df["config_id"] == cfg]
        arch = cfg_rows["arch"].iloc[0]
        is_baseline = arch in BASELINE_ARCHS

        windows_out: "dict[str, dict]" = {}
        n_pass = 0
        n_margin_pass = 0
        any_status_bad = False
        std_fail_windows: "list[str]" = []
        for w in window_ids:
            g = cfg_rows[cfg_rows["window_id"] == w]
            if g.empty:
                windows_out[w] = {
                    "mcc_ok": False,
                    "f1_ok": False,
                    "std_ok": False,
                    "status_ok": False,
                    "notes": ["missing_window"],
                }
                any_status_bad = True
                std_fail_windows.append(w)
                continue
            clause = _window_clause(g, ridge_windows[w], margin_used, std_bound)
            windows_out[w] = clause
            if clause["window_pass"]:
                n_pass += 1
            if clause["mcc_ok"] and clause["f1_ok"]:
                n_margin_pass += 1
            if not clause["status_ok"]:
                any_status_bad = True
            if not clause["std_ok"]:
                std_fail_windows.append(w)

        margin_majority = n_margin_pass > majority_of / 2
        std_every = not std_fail_windows
        status_every = not any_status_bad

        if std_mode == "every":
            viable = bool((not is_baseline) and margin_majority and std_every and status_every)
        else:  # "majority": pre-fix behaviour, std_ok folded into window_pass
            viable = bool((not is_baseline) and (n_pass > majority_of / 2) and status_every)

        entry = {"arch": arch, **windows_out, "viable": viable}
        if is_baseline:
            entry["notes"] = ["baseline: never a viable candidate (data-model.md §GateDecision)"]
        per_config[cfg] = entry

        if not is_baseline and margin_majority and status_every and not std_every:
            stability_only_failures.append(
                {"config_id": cfg, "arch": arch, "windows": std_fail_windows}
            )

    viable_set = [cfg for cfg, v in per_config.items() if v["viable"]]

    notes = [
        "RIDGE/MAJORITY/PERSISTENCE/ROCKET are baselines: shown for reference, never in viable_set.",
        "margin_used = max(margin_floor, 2 x RIDGE's pooled across-seed MCC std); windows where "
        "RIDGE itself had <2 seeds are excluded from that pooled std (research R8/FR-028).",
        "A config with a missing, collapsed or failed window is never viable, regardless of its "
        "metrics on the other windows.",
        f"std_mode={std_mode}: 'every' (spec FR-028(ii)) requires std_ok on EVERY window "
        "independent of the margin-majority vote; 'majority' folds std_ok into window_pass "
        "so instability can be outvoted (pre-fix behaviour, kept for comparison only).",
        "stability_only_failures lists configs that clear the margin on a majority of windows "
        "and have no bad status, but fail the stability bound on at least one window -- FR-004 "
        "allows these to receive a third seed before a final fail decision.",
    ]

    return {
        "phase": phase,
        "rule": {
            "margin_floor": margin_floor,
            "margin_used": margin_used,
            "std_bound": std_bound,
            "majority_of": majority_of,
            "std_mode": std_mode,
        },
        "ridge_reference": {
            **{w: {"mcc": v["mcc"], "macro_f1": v["macro_f1"]} for w, v in ridge_windows.items()},
            "pooled_seed_std_mcc": ridge_pooled_seed_std_mcc,
        },
        "per_config": per_config,
        "viable_set": viable_set,
        "stability_only_failures": stability_only_failures,
        "decided_at": datetime.now(timezone.utc).isoformat(),
        "decided_by": "gate.py + author review at checkpoint",
        "notes": notes,
    }


def _fmt(x, nd: int = 3) -> str:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "NaN"
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def _render_markdown(decision: dict) -> str:
    window_ids = sorted(k for k in decision["ridge_reference"] if k != "pooled_seed_std_mcc")
    lines = [
        f"# Gate decision: phase {decision['phase']}",
        "",
        "Classification metrics against fixed gbb labels; NOT evidence of trading edge "
        "(Constitution III).",
        "",
        "## Rule (FR-028)",
        "",
        f"- margin_floor = {decision['rule']['margin_floor']}",
        f"- margin_used = {_fmt(decision['rule']['margin_used'])}",
        f"- std_bound = {decision['rule']['std_bound']}",
        f"- majority_of = {decision['rule']['majority_of']}",
        f"- std_mode = {decision['rule']['std_mode']}",
        "",
        "## RIDGE reference",
        "",
        "| window | MCC (seed-mean) | macro-F1 (seed-mean) |",
        "|---|---|---|",
    ]
    for w in window_ids:
        ref = decision["ridge_reference"][w]
        lines.append(f"| {w} | {_fmt(ref['mcc'])} | {_fmt(ref['macro_f1'])} |")
    lines.append(
        f"\npooled_seed_std_mcc = {_fmt(decision['ridge_reference']['pooled_seed_std_mcc'])}\n"
    )

    lines += ["## Per-config", "", "| config_id | arch | " + " | ".join(window_ids) + " | viable |"]
    lines.append("|" + "---|" * (3 + len(window_ids)))
    for cfg, entry in decision["per_config"].items():
        marks = []
        for w in window_ids:
            wclause = entry.get(w, {})
            marks.append("PASS" if wclause.get("window_pass") else "fail")
        lines.append(
            f"| {cfg[:12]} | {entry['arch']} | " + " | ".join(marks) + f" | {entry['viable']} |"
        )

    lines += ["", "## Viable set", ""]
    if decision["viable_set"]:
        for cfg in decision["viable_set"]:
            lines.append(f"- {cfg}")
    else:
        lines.append("_none viable_")

    lines += ["", "## Stability-only failures (third-seed candidates)", ""]
    if decision["stability_only_failures"]:
        lines.append("| config_id | arch | failing windows |")
        lines.append("|---|---|---|")
        for item in decision["stability_only_failures"]:
            lines.append(
                f"| {item['config_id'][:12]} | {item['arch']} | "
                f"{', '.join(item['windows'])} |"
            )
    else:
        lines.append("_none_")

    lines += ["", "## Notes", ""]
    for n in decision["notes"]:
        lines.append(f"- {n}")
    lines += ["", f"decided_at: {decision['decided_at']}", f"decided_by: {decision['decided_by']}"]

    return "\n".join(lines) + "\n"


def write_decision(
    phase: int,
    margin_floor: float = 0.02,
    std_bound: float = 0.03,
    majority_of: Optional[int] = None,
    results_path: "str | Path" = RESULTS_PATH,
    out_dir: "str | Path" = GATES_DIR,
    std_mode: str = "every",
) -> dict:
    """`decide()` + write `gates/phase_N.{json,md}`; returns the decision dict."""
    decision = decide(
        phase,
        margin_floor=margin_floor,
        std_bound=std_bound,
        majority_of=majority_of,
        results_path=results_path,
        std_mode=std_mode,
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"phase_{phase}.json").write_text(json.dumps(decision, indent=2, sort_keys=False))
    (out_dir / f"phase_{phase}.md").write_text(_render_markdown(decision))
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description="research.clfstudy gate")
    parser.add_argument("--phase", type=int, required=True)
    parser.add_argument("--margin-floor", type=float, default=0.02)
    parser.add_argument("--std-bound", type=float, default=0.03)
    parser.add_argument(
        "--std-mode",
        choices=["every", "majority"],
        default="every",
        help="'every' (spec FR-028(ii), default) requires the stability bound on every window; "
        "'majority' reproduces the pre-fix behaviour of folding std_ok into the per-window "
        "majority vote.",
    )
    args = parser.parse_args()
    decision = write_decision(
        args.phase,
        margin_floor=args.margin_floor,
        std_bound=args.std_bound,
        std_mode=args.std_mode,
    )
    print(f"margin_used: {decision['rule']['margin_used']:.4f}")
    print("viable_set:")
    for cfg in decision["viable_set"]:
        print(f"  {cfg}")
    if not decision["viable_set"]:
        print("  (none)")
    print("stability_only_failures (third-seed candidates):")
    for item in decision["stability_only_failures"]:
        print(f"  {item['config_id']} ({item['arch']}): windows={item['windows']}")
    if not decision["stability_only_failures"]:
        print("  (none)")


if __name__ == "__main__":
    main()
