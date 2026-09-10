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


def trial_id(objective: str, control: str, gamma: float, seed: int, fold: str,
             *, leaky: bool = False) -> str:
    tag = "leaky-" if leaky else ""
    return f"{tag}{objective}-{control}-g{gamma}-s{seed}-{fold}"


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


def complete(tid: str, result: dict, path: Path = LOG) -> None:
    """Append the outcome for a previously pre-registered trial."""
    with path.open("a") as f:
        f.write(json.dumps({"trial_id": tid, "completed_at": _now(),
                            "result": result}, sort_keys=True) + "\n")


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
    if len(ids) != len(set(ids)):
        problems.append("duplicate trial_id among pre-registered records")
    for r in pre:
        if r.get("result") is not None:
            problems.append(f"{r['trial_id']}: pre-registered record carried a result")
    pre_ids = set(ids)
    for r in done:
        if r["trial_id"] not in pre_ids:
            problems.append(f"{r['trial_id']}: result without a pre-registered record")
    return not problems, {
        "n_pre_registered": len(pre),
        "n_completed": len(done),
        "n_unique": len(set(ids)),
        "trial_count_for_dsr_pbo": len(pre),
        "problems": problems,
    }
