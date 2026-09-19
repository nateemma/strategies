# pragma pylint: disable=C0103, C0114, C0115, C0116, C0301, C0303, C0411, C0413
# pylint: disable=import-error
# flake8: noqa: F401, E402

"""
Report: assemble `report/report.md` from existing artefacts (T061, data-model.md §Report,
spec FR-035, SC-009).

`assemble()` reads `report/methods.md`, `report/calibration.md`, `report/leaderboard_<N>.md`
(one per phase present), `gates/phase_<N>.md` and its matching `gates/review_<N>*.md` memos,
`specs/018-nnc-classifier-study/quickstart.md`, and `results/results.jsonl` (via
`leaderboard.load_results`, the same dedupe-by-last-line rule as the rest of the harness), and
concatenates them into one markdown document with the section skeleton FR-035 requires:
scope & methods, leaderboards (with plot links), gate decisions (with review memos as
sub-sections), a trigger-rule reference table computed from `results.jsonl`, narrative
placeholders for the recommendation / label-concern / open-question sections (populated with
whatever "Decision"/"Recommendations"/"Open questions" headings can be found in the review
memos, else left as an empty `<!-- NARRATIVE: ... -->` stub for the author), and a
reproduction-commands appendix.

Every artefact is included VERBATIM (no numbers are computed here except the trigger-rule
table, which is a plain mean/groupby over `results.jsonl` columns that already exist). This
module never computes or reports anything about backtest P&L, trades or drawdown (FR-036);
`scan_for_forbidden_terms()`/`--scan` is the SC-009 check that nothing has crept in.

Every artefact path is a parameter with a module-constant default, so tests can point the
whole assembly at a tmp directory; a missing file is skipped (a short note is left in its
place) rather than raising, per T061/T062.
"""
from __future__ import annotations

import argparse
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from research.clfstudy import GATES_DIR, PKG, REPORT_DIR, RESULTS_PATH, ROOT
from research.clfstudy.leaderboard import load_results

#: reproduced from spec.md's Scope Statement / Constitution III.
PRINCIPLE_III_NOTE = (
    "This report measures **classifiers**, not strategies. Every number in it is a "
    "classification metric computed against the fixed, pre-existing ternary gbb labels "
    "(SELL=0, HOLD=1, BUY=2). Learnability is not evidence of trading edge (Constitution "
    "III): these are screening results that select candidate architectures and inputs for "
    "a later, separate, P&L-based evaluation; nothing here promotes anything to production."
)

#: `specs/018-nnc-classifier-study/quickstart.md` (T061's reproduction-commands appendix).
QUICKSTART_PATH = ROOT / "specs" / "018-nnc-classifier-study" / "quickstart.md"

#: SC-009's forbidden-term pattern (backtest/P&L/drawdown/trade-level language anywhere but
#: the scope statement / label-concern notes).
FORBIDDEN_PATTERN = re.compile(r"backtest|p&l|pnl|drawdown|trade-level", re.IGNORECASE)


def _git_sha() -> str:
    """Short git HEAD sha, `"unknown"` if git is unavailable (mirrors `run.py`'s helper)."""
    try:
        result = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        if result.returncode == 0:
            sha = result.stdout.strip()
            if sha:
                return sha
    except Exception:
        pass
    return "unknown"


def _read_text(path: Path) -> Optional[str]:
    """`path`'s text, or `None` if it doesn't exist / can't be read (never raises)."""
    try:
        if path.is_file():
            return path.read_text(encoding="utf-8")
    except OSError:
        pass
    return None


def _missing_note(path: Path) -> str:
    return f"_Not found: `{path}`._"


def _heading_level(line: str) -> int:
    return len(line) - len(line.lstrip("#"))


def _slice_between_headings(lines: list, start_prefix: str, end_prefix: str) -> Optional[list]:
    """Lines from the first line starting with `start_prefix` up to (excluding) the first
    later line starting with `end_prefix`; to EOF if `end_prefix` isn't found; `None` if
    `start_prefix` isn't found at all."""
    start = next((i for i, l in enumerate(lines) if l.startswith(start_prefix)), None)
    if start is None:
        return None
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith(end_prefix)), len(lines))
    return lines[start:end]


def extract_leaderboard_tables(leaderboard_md_text: str) -> str:
    """Table 1 + Table 2 (everything between the `## Table 1` and `## Table 3` headings) of
    one `leaderboard_<phase>.md` (FR-035: "ranked leaderboard"). Verbatim other than the slice."""
    lines = leaderboard_md_text.splitlines()
    sliced = _slice_between_headings(lines, "## Table 1", "## Table 3")
    if sliced is None:
        return leaderboard_md_text.rstrip("\n")
    return "\n".join(sliced).rstrip("\n")


def _extract_headed_subsections(text: str, keyword_regex: str) -> list:
    """Every markdown subsection of `text` whose heading line matches `keyword_regex`
    (case-insensitive), each returned as its heading line through to (excluding) the next
    heading of equal-or-shallower level, or EOF."""
    lines = text.splitlines()
    out = []
    for i, line in enumerate(lines):
        if not line.startswith("#") or not re.search(keyword_regex, line, re.IGNORECASE):
            continue
        level = _heading_level(line)
        end = len(lines)
        for j in range(i + 1, len(lines)):
            if lines[j].startswith("#") and _heading_level(lines[j]) <= level:
                end = j
                break
        out.append("\n".join(lines[i:end]).rstrip("\n"))
    return out


def _discover_phases(report_dir: Path) -> list:
    """Phase numbers with a `leaderboard_<N>.md` under `report_dir`, sorted ascending."""
    phases = []
    for p in Path(report_dir).glob("leaderboard_*.md"):
        m = re.fullmatch(r"leaderboard_(\d+)\.md", p.name)
        if m:
            phases.append(int(m.group(1)))
    return sorted(phases)


def _discover_gate_phases(gates_dir: Path) -> list:
    """Phase numbers with a `gates/phase_<N>.md`, sorted ascending."""
    phases = []
    for p in Path(gates_dir).glob("phase_*.md"):
        m = re.fullmatch(r"phase_(\d+)\.md", p.name)
        if m:
            phases.append(int(m.group(1)))
    return sorted(phases)


def _matching_review_files(gates_dir: Path, phase: int) -> list:
    """`review_<phase>*.md` under `gates_dir`, sorted by filename."""
    return sorted(Path(gates_dir).glob(f"review_{phase}*.md"))


def _fmt(x, nd: int = 3) -> str:
    try:
        if x is None:
            return "NaN"
        xf = float(x)
        if xf != xf:  # NaN
            return "NaN"
        return f"{xf:.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def build_trigger_rule_section(results_path: Path) -> str:
    """Trigger-rule reference table (T061 step 5): the learning-free rule's per-window MCC
    (mean of `trig_rule_mcc` over every row that carries it, any phase) and, for phase 3/4
    rows only, the within-trigger BUY AUROC and precision lift
    (`trig_buy_prec_argmax - trig_buy_base_rate`) per arch per window.
    """
    df = load_results(results_path)
    lines = [
        "The learning-free rule: BUY/SELL if `guard_metric` is on the appropriate side of its "
        "threshold and `bb_width` clears its threshold, else HOLD (methods.md, "
        "\"The ceiling is the label's trigger rule\"). `trig_rule_mcc` reproduces this rule's "
        "MCC against the same test rows a trained config scored against; `trig_buy_*` are "
        "computed only within rows where the rule fired BUY.",
        "",
    ]

    if df.empty or "trig_rule_mcc" not in df.columns:
        lines.append("_No `trig_*` rows found in `results.jsonl`._")
        return "\n".join(lines)

    rule = df[df["trig_rule_mcc"].notna()]
    lines += [
        "### Trigger-rule MCC per window",
        "",
        "| window | trig_rule_mcc (mean) | n rows |",
        "|---|---|---|",
    ]
    if rule.empty:
        lines.append("| _none_ | - | 0 |")
    else:
        for window_id, g in rule.groupby("window_id", sort=True):
            lines.append(f"| {window_id} | {_fmt(g['trig_rule_mcc'].mean())} | {len(g)} |")

    lines += ["", "### Within-trigger BUY metrics per arch per window (phase 3 and 4 rows)", ""]
    lines += ["| phase | arch | window | trig_buy_auroc (mean) | precision lift (mean) | n rows |",
              "|---|---|---|---|---|---|"]
    buy_cols = {"trig_buy_auroc", "trig_buy_prec_argmax", "trig_buy_base_rate"}
    if not buy_cols.issubset(df.columns):
        lines.append("| _none_ | - | - | - | - | 0 |")
        return "\n".join(lines)

    sub = df[df["phase"].isin([3, 4]) & df["trig_buy_auroc"].notna()].copy()
    if sub.empty:
        lines.append("| _none_ | - | - | - | - | 0 |")
    else:
        sub["precision_lift"] = sub["trig_buy_prec_argmax"] - sub["trig_buy_base_rate"]
        for (phase, arch, window_id), g in sub.groupby(["phase", "arch", "window_id"], sort=True):
            lines.append(
                f"| {int(phase)} | {arch} | {window_id} | {_fmt(g['trig_buy_auroc'].mean())} "
                f"| {_fmt(g['precision_lift'].mean())} | {len(g)} |"
            )

    return "\n".join(lines)


def build_leaderboards_section(report_dir: Path) -> str:
    """FR-035's "ranked leaderboard": Table 1 + Table 2 of every `leaderboard_<phase>.md`
    present, followed by links to that phase's plots (`plots/` under `report_dir`)."""
    report_dir = Path(report_dir)
    phases = _discover_phases(report_dir)
    if not phases:
        return "_No `leaderboard_*.md` files found._"

    plot_kinds = ("mcc_bar", "mcc_heatmap", "mcc_vs_params", "reliability")
    parts = []
    for phase in phases:
        parts.append(f"### Phase {phase}")
        parts.append("")
        text = _read_text(report_dir / f"leaderboard_{phase}.md")
        if text is None:
            parts.append(_missing_note(report_dir / f"leaderboard_{phase}.md"))
        else:
            parts.append(extract_leaderboard_tables(text))
        parts.append("")
        plot_links = [
            f"![{kind} phase {phase}](plots/{kind}_{phase}.png)"
            for kind in plot_kinds
            if (report_dir / "plots" / f"{kind}_{phase}.png").is_file()
        ]
        if plot_links:
            parts.append("\n".join(plot_links))
        else:
            parts.append("_No plots found for this phase._")
        parts.append("")
    return "\n".join(parts).rstrip("\n")


def build_gates_section(gates_dir: Path) -> str:
    """FR-035's "every gate decision": verbatim `gates/phase_<N>.md` for each phase present,
    followed by its matching `review_<N>*.md` memo(s) as sub-sections."""
    gates_dir = Path(gates_dir)
    phases = _discover_gate_phases(gates_dir)
    if not phases:
        return "_No `gates/phase_*.md` files found._"

    parts = []
    for phase in phases:
        parts.append(f"### Phase {phase} gate decision")
        parts.append("")
        text = _read_text(gates_dir / f"phase_{phase}.md")
        parts.append(text if text is not None else _missing_note(gates_dir / f"phase_{phase}.md"))
        parts.append("")
        for review_path in _matching_review_files(gates_dir, phase):
            review_text = _read_text(review_path)
            parts.append(f"#### Review memo: `{review_path.name}`")
            parts.append("")
            parts.append(review_text if review_text is not None else _missing_note(review_path))
            parts.append("")
    return "\n".join(parts).rstrip("\n")


def build_narrative_section(gates_dir: Path, keyword_regex: str, empty_stub: str) -> str:
    """Subsections matching `keyword_regex` pulled from every `gates/review_*.md`, prefixed
    by the source filename; `empty_stub` (a `<!-- NARRATIVE: ... -->` placeholder) if none
    are found."""
    gates_dir = Path(gates_dir)
    found = []
    for review_path in sorted(gates_dir.glob("review_*.md")):
        text = _read_text(review_path)
        if text is None:
            continue
        for sub in _extract_headed_subsections(text, keyword_regex):
            found.append(f"_From `{review_path.name}`:_\n\n{sub}")
    if not found:
        return empty_stub
    return "\n\n".join(found)


def build_reproduction_commands_section(quickstart_path: Path) -> str:
    """FR-035's reproduction-commands appendix: the fenced bash block of `quickstart.md`,
    verbatim."""
    text = _read_text(Path(quickstart_path))
    if text is None:
        return _missing_note(Path(quickstart_path))
    m = re.search(r"```bash\n(.*?)```", text, re.DOTALL)
    if not m:
        return text.rstrip("\n")
    return "```bash\n" + m.group(1) + "```"


def assemble(
    report_dir: Path = REPORT_DIR,
    gates_dir: Path = GATES_DIR,
    results_path: Path = RESULTS_PATH,
    quickstart_path: Path = QUICKSTART_PATH,
) -> str:
    """Build the full `report.md` text (data-model.md §Report, FR-035). Never raises on a
    missing artefact -- each section reports "not found" in its place instead (T061/T062)."""
    report_dir = Path(report_dir)
    gates_dir = Path(gates_dir)

    methods_text = _read_text(report_dir / "methods.md")
    calibration_text = _read_text(report_dir / "calibration.md")

    parts = [
        "# NNC Classifier Study Report",
        "",
        PRINCIPLE_III_NOTE,
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}, git sha `{_git_sha()}`",
        "",
        "## 1. Scope & methods",
        "",
        methods_text if methods_text is not None else _missing_note(report_dir / "methods.md"),
        "",
        "### Calibration",
        "",
        calibration_text if calibration_text is not None else _missing_note(report_dir / "calibration.md"),
        "",
        "## 2. Leaderboards",
        "",
        build_leaderboards_section(report_dir),
        "",
        "## 3. Gate decisions",
        "",
        build_gates_section(gates_dir),
        "",
        "## 4. Trigger-rule reference",
        "",
        build_trigger_rule_section(results_path),
        "",
        "## 5. Recommendation",
        "",
        "<!-- NARRATIVE: author fills in the recommended architecture(s) and input set with "
        "the evidence rows that support each (SC-010); the extracts below are pulled from the "
        "review memos' Decision/Recommendations sections and are a starting point, not a "
        "final answer. -->",
        "",
        build_narrative_section(
            gates_dir,
            r"decision|recommend",
            "<!-- NARRATIVE: no Decision/Recommendations heading found in any review memo. -->",
        ),
        "",
        "## 6. Label concerns",
        "",
        "<!-- NARRATIVE: author summarises the label-definition concerns (research R11) -- "
        "see methods.md \"The ceiling is the label's trigger rule\" above: every trained "
        "architecture converges to the learning-free trigger rule's MCC and nothing beyond "
        "it, which is a property of the gbb label at this input set/horizon, not of any "
        "architecture. -->",
        "",
        "## 7. Open questions",
        "",
        build_narrative_section(
            gates_dir,
            r"open questions?",
            "<!-- NARRATIVE: no Open questions heading found in any review memo. -->",
        ),
        "",
        "## 8. Reproduction commands",
        "",
        build_reproduction_commands_section(quickstart_path),
        "",
    ]
    return "\n".join(parts) + "\n"


def write_report(
    report_dir: Path = REPORT_DIR,
    gates_dir: Path = GATES_DIR,
    results_path: Path = RESULTS_PATH,
    quickstart_path: Path = QUICKSTART_PATH,
) -> Path:
    """`assemble()` then write it to `<report_dir>/report.md`; returns that path."""
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    text = assemble(report_dir, gates_dir, results_path, quickstart_path)
    out_path = report_dir / "report.md"
    out_path.write_text(text, encoding="utf-8")
    return out_path


def scan_for_forbidden_terms(report_path: Optional[Path] = None, package_dir: Path = PKG) -> list:
    """SC-009: every (file, line number, line) in `report_path` (if it exists) and every
    `*.py` under `package_dir` matching `FORBIDDEN_PATTERN`. Never raises."""
    targets = []
    if report_path is not None and Path(report_path).is_file():
        targets.append(Path(report_path))
    targets.extend(sorted(Path(package_dir).rglob("*.py")))

    hits = []
    for path in targets:
        text = _read_text(path)
        if text is None:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if FORBIDDEN_PATTERN.search(line):
                hits.append((str(path), lineno, line.strip()))
    return hits


def main() -> None:
    parser = argparse.ArgumentParser(description="research.clfstudy report")
    parser.add_argument("--report-dir", type=str, default=str(REPORT_DIR))
    parser.add_argument("--gates-dir", type=str, default=str(GATES_DIR))
    parser.add_argument("--results", type=str, default=str(RESULTS_PATH))
    parser.add_argument("--quickstart", type=str, default=str(QUICKSTART_PATH))
    parser.add_argument(
        "--scan", action="store_true",
        help="SC-009: grep the report and the package's *.py for forbidden terms; print, don't fail",
    )
    args = parser.parse_args()

    if args.scan:
        report_path = Path(args.report_dir) / "report.md"
        hits = scan_for_forbidden_terms(report_path, PKG)
        if not hits:
            print("scan: no hits")
        for path, lineno, line in hits:
            print(f"{path}:{lineno}: {line}")
        return

    out_path = write_report(
        report_dir=args.report_dir,
        gates_dir=args.gates_dir,
        results_path=args.results,
        quickstart_path=args.quickstart,
    )
    print(f"wrote: {out_path}")
    for line in out_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            print(f"  {line}")


if __name__ == "__main__":
    main()
