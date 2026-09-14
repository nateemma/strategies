"""Persisted path banks, with the provenance to detect staleness (T004, T005, T019).

WHY PROVENANCE EXISTS. A bank generated before a panel changes would silently stress the wrong
thing. This is not hypothetical: feature 012 recorded that a cache key must cover everything
that changes the panel, and feature 013's panel cache keys on the side-car's content hash for
exactly that reason. `load()` RECOMPUTES the source hash and REFUSES a moved bank rather than
warning about it.

SAMPLING IS OFFLINE AND ONCE (FR-001, SC-005). A bank regenerated per run is not reproducible,
which defeats the point even though nothing here is slow enough to need a cache.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.labelfree.es import panel as PA                    # noqa: E402
from research.labelfree.stress import bootstrap as BS            # noqa: E402

BANKS = Path(__file__).resolve().parent / "banks"
GENERATOR = "stationary_block_bootstrap"


def source_hash(panels: dict) -> str:
    """Hash of every panel feeding a bank — the staleness detector (FR-002)."""
    h = hashlib.sha256()
    for fold in sorted(panels):
        for pair in sorted(panels[fold]["pairs"]):
            d = panels[fold]["pairs"][pair]
            h.update(fold.encode())
            h.update(pair.encode())
            for col in BS.PANEL_COLUMNS:
                if col in d and d[col] is not None:
                    h.update(np.ascontiguousarray(d[col], dtype=np.float64).tobytes())
    return h.hexdigest()


def provenance(fold_ids, block_len, seed, n_paths, panels) -> dict:
    cols = sorted({c for f in panels.values() for d in f["pairs"].values()
                   for c in d if c in BS.PANEL_COLUMNS})
    return {"generator": GENERATOR, "block_length": int(block_len), "seed": int(seed),
            "fold_ids": list(fold_ids), "n_paths": int(n_paths),
            "pairs": sorted({p for f in panels.values() for p in f["pairs"]}),
            "feature_columns": cols,
            "source_sha256": source_hash(panels),
            "shared_block_index": True,
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def bank_path(fold_ids, block_len, seed) -> Path:
    tag = f"{fold_ids[0]}-{fold_ids[-1]}" if len(fold_ids) > 1 else fold_ids[0]
    return BANKS / f"{tag}_L{block_len}_s{seed}.npz"


def build(fold_ids, block_len: int, n_paths: int, seed: int,
          out: Path | None = None) -> Path:
    panels = {f: PA.load(f, train=False) for f in fold_ids}
    prov = provenance(fold_ids, block_len, seed, n_paths, panels)

    # One index vector per (fold, path). Storing INDICES rather than resampled arrays keeps the
    # bank small and makes it exactly reproducible from the source panels.
    rng = np.random.default_rng(seed)
    idx = {}
    for f in fold_ids:
        n = max(len(d["open"]) for d in panels[f]["pairs"].values())
        idx[f] = np.stack([BS.block_index(n, n, block_len, rng) for _ in range(n_paths)])

    out = out or bank_path(fold_ids, block_len, seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, provenance=json.dumps(prov),
                        **{f"idx_{f}": v for f, v in idx.items()})
    return out


def load(path: Path, verify: bool = True) -> tuple[dict, dict, dict]:
    """(provenance, {fold: index matrix}, {fold: live panel}).

    REFUSES a bank whose source panels have moved — a stale bank stresses the wrong thing and
    would do it silently.
    """
    z = np.load(path, allow_pickle=False)
    prov = json.loads(str(z["provenance"]))
    panels = {f: PA.load(f, train=False) for f in prov["fold_ids"]}
    if verify:
        now = source_hash(panels)
        if now != prov["source_sha256"]:
            raise SystemExit(
                f"STALE BANK: {path.name} was built from source {prov['source_sha256'][:12]}, "
                f"the panels now hash {now[:12]}. Rebuild it — a bank whose source has moved "
                f"stresses the wrong thing, and would do so silently.")
    idx = {f: z[f"idx_{f}"] for f in prov["fold_ids"]}
    return prov, idx, panels


def paths(prov: dict, idx: dict, panels: dict):
    """Yield (fold_id, resampled panel) for every stored path."""
    for f in prov["fold_ids"]:
        for row in idx[f]:
            yield f, BS.resample_panel(panels[f], row)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--folds", default="f05..f14")
    ap.add_argument("--block", type=int, default=192)
    ap.add_argument("--paths", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    from research.labelfree.baselines import parse_span
    folds = parse_span(a.folds)
    if not a.build:
        ap.error("nothing to do; pass --build")
    t0 = time.time()
    out = build(folds, a.block, a.paths, a.seed)
    prov = json.loads(str(np.load(out, allow_pickle=False)["provenance"]))
    print(f"{out}  ({out.stat().st_size / 1e6:.1f} MB, {time.time() - t0:.0f}s)")
    print(f"  folds {len(prov['fold_ids'])}  pairs {len(prov['pairs'])}  "
          f"paths {prov['n_paths']}  L={prov['block_length']}  seed {prov['seed']}")
    print(f"  source sha256 {prov['source_sha256'][:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
