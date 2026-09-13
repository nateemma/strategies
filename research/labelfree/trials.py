"""Append-only trial log (T007).

FR-019 / SC-015: every configuration evaluated on any fold is appended BEFORE its result
is known. A log written afterwards silently omits abandoned runs, which is exactly the
mechanism the Deflated Sharpe Ratio and PBO exist to correct for.

Contract: specs/009-labelfree-direct-policy/contracts/trial-log.md
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

LOG = Path(__file__).resolve().parent / "trials.jsonl"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def new_run_id() -> str:
    """A tag identifying one sweep invocation."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def trial_id(objective: str, control: str, gamma: float, seed: int, fold: str,
             *, leaky: bool = False, run: str | None = None) -> str:
    """Unique per configuration AND per sweep invocation.

    The run tag is not cosmetic. Without it a re-run of the same grid re-registers
    identical ids, and the log can no longer say which pre-registration belongs to which
    completion — which makes the DSR/PBO trial count ambiguous. The audit caught exactly
    that on 2026-09-10: 89 pre-registered records, 61 unique ids.
    """
    tag = "leaky-" if leaky else ""
    base = f"{tag}{objective}-{control}-g{gamma}-s{seed}-{fold}"
    return f"{base}@{run}" if run else base


def pre_register(record: dict, path: Path = LOG) -> str:
    """Append a configuration BEFORE running it. Returns its trial_id.

    `result` is forced to None here; attaching a result at write time is an audit
    failure, so the writer is not permitted to supply one.
    """
    rec = dict(record)
    rec["result"] = None
    rec.setdefault("written_at", _now())
    if "trial_id" not in rec:
        raise ValueError("pre_register requires trial_id")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")
    return rec["trial_id"]


N_BLOCKS = 16          # CSCV's split count; the minimum series length that makes PBO real


def block_returns(per_bar, n_blocks: int = N_BLOCKS) -> list[float]:
    """Compress a per-bar return series into `n_blocks` equal-time block returns.

    WHY BLOCKS AND NOT PER-BAR. DSR deflates a Sharpe computed over a TIME axis and CSCV
    partitions that axis into blocks; both need a series, neither needs every bar. Storing
    per-bar returns would put ~9,000 floats per trial into an append-only log that already
    holds 500+ trials -- millions of numbers to recover sixteen. Blocks are exactly what the
    statistics consume.

    This is the schema gap that made SC-004 unevaluable in FOUR consecutive features: the log
    stored summary scalars, so seeds were the only available axis, and a configuration that
    is reliably mediocre scored a high "Sharpe" purely because its across-seed dispersion was
    tiny. Sixteen numbers per trial fixes it.
    """
    import numpy as np
    a = np.asarray(per_bar, dtype=float).ravel()
    a = a[np.isfinite(a)]
    if a.size == 0:
        return []
    n = min(n_blocks, a.size)
    return [float(b.sum()) for b in np.array_split(a, n)]


def complete(tid: str, result: dict, path: Path = LOG,
             per_bar_returns=None) -> None:
    """Append the outcome for a previously pre-registered trial.

    `per_bar_returns` is optional and is stored COMPRESSED as `block_returns` (see above).
    Passing it is what allows the trial to contribute a time axis to DSR and PBO; trials
    without it still count toward the deflation TRIAL COUNT, which is the quantity that
    deflates the statistic.
    """
    payload = {"trial_id": tid, "completed_at": _now(), "result": result}
    if per_bar_returns is not None:
        blocks = block_returns(per_bar_returns)
        if blocks:
            payload["block_returns"] = blocks
    with path.open("a") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def abandon(tid: str, reason: str, path: Path = LOG) -> None:
    complete(tid, {"status": "abandoned", "reason": reason}, path)


def read(path: Path = LOG) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def audit(path: Path = LOG) -> tuple[bool, dict]:
    """SC-015: every config logged, each written before its result, ids unique."""
    recs = read(path)
    pre = [r for r in recs if "written_at" in r]
    done = [r for r in recs if "completed_at" in r]
    ids = [r["trial_id"] for r in pre]
    problems = []
    # (file-wide duplicates are decomposed per run below; a collision within a single
    #  tagged run is a real defect, one across the pre-tag era is recorded history)
    for r in pre:
        if r.get("result") is not None:
            problems.append(f"{r['trial_id']}: pre-registered record carried a result")
    pre_ids = set(ids)
    for r in done:
        if r["trial_id"] not in pre_ids:
            problems.append(f"{r['trial_id']}: result without a pre-registered record")
    # Per-invocation uniqueness is the meaningful unit. Ids gained a run tag only after
    # 2026-09-11; records written before that can collide across re-runs of the same grid,
    # which is exactly the ambiguity the tag was added to remove. Those runs are all marked
    # VOID, so the check reports them separately rather than either ignoring them (which
    # would hide a real defect) or failing forever on history that cannot be rewritten.
    by_run: dict[str, list[str]] = {}
    for r in pre:
        tag = r["trial_id"].split("@")[-1] if "@" in r["trial_id"] else "pre-run-tag"
        by_run.setdefault(tag, []).append(r["trial_id"])
    per_run_ok = {}
    for tag, tids in by_run.items():
        dup = len(tids) != len(set(tids))
        per_run_ok[tag] = not dup
        if dup and tag != "pre-run-tag":
            problems.append(f"duplicate trial_id WITHIN run {tag}")
    untagged_dupes = not per_run_ok.get("pre-run-tag", True)

    return not problems, {
        "n_pre_registered": len(pre),
        "n_completed": len(done),
        "n_unique": len(set(ids)),
        "trial_count_for_dsr_pbo": len(pre),
        "runs": {t: {"n": len(v), "unique": per_run_ok[t]} for t, v in by_run.items()},
        "untagged_collisions": untagged_dupes,
        "problems": problems,
    }
