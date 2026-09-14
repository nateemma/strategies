"""The per-feature sidecar evaluation log (T008), and its audit (T041).

WHY A SIDECAR. The shared trials.jsonl holds 1,158 lines for the 505 trials of four completed
features and is committed to git. This feature implies up to 48,000 evaluations -- roughly two
orders of magnitude more than the study's entire history, in a file every future feature's
audit would have to read. So the full history lives here and the shared log gets ONE SUMMARY
ROW per search run, carrying this file's line count and sha256.

The obvious failure that buys is an audit certifying a record it never examined, which is
worse than the storage problem. `audit()` therefore checks BOTH files (SC-018).

WRITE BEFORE SCORE (FR-013). `write()` appends with score: null; `complete()` fills it in.
A crash between the two leaves evidence of an ATTEMPT, which is exactly what the deflation
must count -- a log that only records successes undercounts attempts and inflates every
downstream statistic.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
SIDECAR = HERE / "history.jsonl"
SHARED = HERE.parent / "trials.jsonl"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write(candidate_id: str, params: dict, fold_id: str, range_: str, method: str,
          generation: int = 0, path: Path = SIDECAR) -> str:
    """Append an evaluation BEFORE it is scored. Returns its eval_id."""
    if range_ not in ("train", "test"):
        raise ValueError(f"range must be 'train' or 'test', got {range_!r}")
    path.parent.mkdir(parents=True, exist_ok=True)
    n = sum(1 for _ in path.open()) if path.exists() else 0
    eval_id = f"e{n:07d}"
    row = {"eval_id": eval_id, "candidate_id": candidate_id, "params": params,
           "fold_id": fold_id, "range": range_, "method": method,
           "generation": generation, "written_at": _now(), "score": None}
    with path.open("a") as fh:
        fh.write(json.dumps(row) + "\n")
    return eval_id


def complete(eval_id: str, result: dict, path: Path = SIDECAR) -> None:
    """Attach a score to an already-written evaluation.

    Appended as a separate record rather than rewriting the line: an append-only file cannot
    lose the original attempt, and the audit reads the pair.
    """
    with path.open("a") as fh:
        fh.write(json.dumps({"eval_id": eval_id, "completed_at": _now(),
                             **result}) + "\n")


def read(path: Path = SIDECAR) -> list[dict]:
    """Merged view: one dict per evaluation, completions folded onto their write."""
    if not path.exists():
        return []
    rows: dict[str, dict] = {}
    order: list[str] = []
    for line in path.open():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        eid = r["eval_id"]
        if eid not in rows:
            rows[eid] = r
            order.append(eid)
        else:
            rows[eid].update(r)
    return [rows[e] for e in order]


def summarise(path: Path = SIDECAR) -> dict:
    """The pointer the shared log carries, so it references the full record (SC-018)."""
    raw = path.read_bytes() if path.exists() else b""
    return {"sidecar": str(path.relative_to(path.parents[2])),
            "lines": raw.count(b"\n"),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "evaluations": len(read(path))}


# ------------------------------------------------------------------------------------ audit

def audit(path: Path = SIDECAR, shared: Path = SHARED) -> tuple[bool, dict]:
    """The leakage proof, from the sidecar ALONE — no trust in the search code (FR-011).

    1. exactly one test-range evaluation per fold                         (SC-011)
    2. every test evaluation later than every train evaluation for it     (FR-010)
    3. every evaluation written before it was scored                      (FR-013)
    4. the shared log's summary row matches this file                     (SC-018)
    """
    rows = read(path)
    problems: list[str] = []

    by_fold: dict[str, dict[str, list[dict]]] = {}
    for r in rows:
        by_fold.setdefault(r["fold_id"], {"train": [], "test": []})[r["range"]].append(r)

    for fold, g in sorted(by_fold.items()):
        if len(g["test"]) != 1:
            problems.append(f"{fold}: {len(g['test'])} test evaluations, expected exactly 1")
            continue
        t = g["test"][0]["written_at"]
        late = [r["eval_id"] for r in g["train"] if r["written_at"] > t]
        if late:
            problems.append(
                f"{fold}: {len(late)} train evaluations written AFTER the test evaluation "
                f"({late[:3]}...) — the search continued past the test range")

    unscored = [r["eval_id"] for r in rows if "completed_at" not in r]
    for r in rows:
        if "completed_at" in r and r["completed_at"] < r["written_at"]:
            problems.append(f"{r['eval_id']}: completed_at precedes written_at")

    shared_ok, shared_note = _check_shared(path, shared)
    if not shared_ok:
        problems.append(shared_note)

    return (not problems), {
        "evaluations": len(rows),
        "folds": len(by_fold),
        "test_evaluations": sum(len(g["test"]) for g in by_fold.values()),
        "abandoned_unscored": len(unscored),
        "shared_log": shared_note,
        "problems": problems,
    }


def _check_shared(path: Path, shared: Path) -> tuple[bool, str]:
    if not shared.exists():
        return False, f"shared trial log {shared} missing — SC-018 needs both files"
    want = summarise(path)
    rows = [json.loads(l) for l in shared.open() if l.strip()]
    refs = [r for r in rows if r.get("sidecar")]
    if not refs:
        return False, ("shared trial log carries no summary row pointing at the sidecar; an "
                       "audit reading only the summary would certify a record it never saw")
    last = refs[-1]
    if last.get("sha256") != want["sha256"] or last.get("lines") != want["lines"]:
        return False, (f"shared summary row is stale: says {last.get('lines')} lines / "
                       f"{str(last.get('sha256'))[:12]}, sidecar has {want['lines']} lines / "
                       f"{want['sha256'][:12]}")
    return True, f"summary row matches ({want['lines']} lines, {want['evaluations']} evaluations)"


if __name__ == "__main__":
    import sys
    ok, rep = audit()
    print(json.dumps(rep, indent=2))
    if not ok:
        print("\nAUDIT FAILED — the run it describes is not admissible.", file=sys.stderr)
    raise SystemExit(0 if ok else 1)
