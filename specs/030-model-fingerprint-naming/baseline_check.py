"""030 T007/T031: compare arm records per fold by trades_hash and n_trades.

    ../../.venv/bin/python specs/030-model-fingerprint-naming/baseline_check.py \
        --pairs wf_aro_b030=wf_aro_a029 sw_fmr_ctl_b030=wf_fmr ... \
        [--single NNMT_MLX_Base_b030:W1=NNMT_MLX_Base_a030:W1 ...] --out baseline.json

A `--pairs` entry is a walk-forward arm (`<tag>_fNN_FNN.json`, f00-f14); a `--single` entry is
one record `<tag>_<WINDOW>.json`. Exit 1 if any compared pair differs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


STRAT = Path(__file__).resolve().parents[2]
ARMS = STRAT / "regime" / "arms"
FOLDS = [f"f{i:02d}" for i in range(15)]


def _rec(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.is_file() else None


def compare_wf(a: str, b: str) -> dict:
    rows = []
    for f in FOLDS:
        ra = _rec(ARMS / f"{a}_{f}_{f.upper()}.json")
        rb = _rec(ARMS / f"{b}_{f}_{f.upper()}.json")
        rows.append({"fold": f,
                     "a": None if ra is None else (ra.get("n_trades"), ra.get("trades_hash")),
                     "b": None if rb is None else (rb.get("n_trades"), rb.get("trades_hash"))})
    eq = sum(r["a"] is not None and r["a"] == r["b"] for r in rows)
    return {"a": a, "b": b, "equal": eq, "of": len(FOLDS), "rows": rows,
            "trades": [sum((r["a"] or (0, 0))[0] or 0 for r in rows),
                       sum((r["b"] or (0, 0))[0] or 0 for r in rows)]}


def compare_single(a: str, b: str) -> dict:
    (ta, wa), (tb, wb) = a.split(":"), b.split(":")
    ra, rb = _rec(ARMS / f"{ta}_{wa}.json"), _rec(ARMS / f"{tb}_{wb}.json")
    ka = None if ra is None else (ra.get("n_trades"), ra.get("trades_hash"))
    kb = None if rb is None else (rb.get("n_trades"), rb.get("trades_hash"))
    return {"a": a, "b": b, "equal": int(ka is not None and ka == kb), "of": 1,
            "rows": [{"fold": wa, "a": ka, "b": kb}],
            "trades": [(ka or (0,))[0], (kb or (0,))[0]]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pairs", nargs="*", default=[])
    ap.add_argument("--single", nargs="*", default=[])
    ap.add_argument("--report-only", nargs="*", default=[],
                    help="pair keys (a=b) whose difference is reported but does not fail")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    res = [compare_wf(*p.split("=")) for p in a.pairs]
    res += [compare_single(*p.split("=")) for p in a.single]
    bad = 0
    for r in res:
        key = f"{r['a']}={r['b']}"
        ok = r["equal"] == r["of"]
        soft = key in a.report_only
        bad += (not ok and not soft)
        print(f"{'OK  ' if ok else ('INFO' if soft else 'DIFF')} {r['a']:<28} vs {r['b']:<28} "
              f"{r['equal']}/{r['of']}  trades {r['trades'][0]} / {r['trades'][1]}")
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
