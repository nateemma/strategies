"""031 T001/T016: framework_sha, model key and recorded trades_hash for the reference arms.

    PYTHONPATH=. ../../.venv/bin/python specs/031-held-pair-dca/baseline.py --out baseline.json
    PYTHONPATH=. ../../.venv/bin/python specs/031-held-pair-dca/baseline.py --check baseline.json \
        --suffix _c031

`--check` recomputes fingerprints and model keys, compares them with the saved baseline, and
compares each reference arm's per-fold trades_hash with the `<tag><suffix>` re-run. Exit 1 on any
difference (and when a `<tag><suffix>` record carries n_adds / hp_counters).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


STRAT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(STRAT))
sys.path.insert(0, str(STRAT / "utils"))

ARMS = STRAT / "regime" / "arms"
FOLDS = STRAT / "evaluation" / "folds"
CONFIG = STRAT / "config" / "config_wf.json"
REFS = ("wf_aro_b030", "wf_aro_g24", "fs_gbr_gr", "fs_gfm_gr", "fs_mlx_gr", "st_bgo_mx",
        "st_prom_c2")
BASES = ("NNNC_A_Raw_Open", "NNNC_GbbRaw_Open", "NNNC_GbbFwdMae_Open", "NNNC_MLX",
         "SimpleTrigger_BreakoutGbb_MomExit", "SimpleTrigger_RevOnlyVolreg2_P2",
         "SimpleTrigger_BreakoutGbb_MomExit_G24", "SimpleTrigger_RevOnlyVolreg2_P2_G24")


def _rec(tag: str, fid: str) -> dict | None:
    p = ARMS / f"{tag}_{fid}_{fid.upper()}.json"
    return json.loads(p.read_text()) if p.is_file() else None


def snapshot() -> dict:
    from evaluation.fingerprint import load_config, load_strategy_class, resolve_by_name
    from Framework import ModelKey

    cfg = load_config(CONFIG)
    out = {"bases": {}, "arms": {}}
    for b in BASES:
        out["bases"][b] = resolve_by_name(b, CONFIG)[0]
    for tag in REFS:
        plan = json.loads((FOLDS / f"plan_{tag}.json").read_text())
        rows = {}
        for f in plan["folds"]:
            cls_name = f.get("strategy_class") or plan["base_strategy"]
            cls = load_strategy_class(cls_name)
            sha = resolve_by_name(cls_name, CONFIG)[0]
            mk = None
            if f.get("model_dir"):
                mk = ModelKey.key_hash(ModelKey.resolve(cls, cfg))
            r = _rec(tag, f["fold_id"]) or {}
            rows[f["fold_id"]] = {"class": cls_name, "framework_sha": sha, "model_key": mk,
                                  "trades_hash": r.get("trades_hash"),
                                  "n_trades": r.get("n_trades")}
        out["arms"][tag] = rows
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out")
    ap.add_argument("--check")
    ap.add_argument("--suffix", default="_c031")
    a = ap.parse_args(argv)
    now = snapshot()
    now["commit"] = subprocess.check_output(["git", "-C", str(STRAT), "rev-parse", "--short",
                                             "HEAD"], text=True).strip()
    if a.out:
        Path(a.out).write_text(json.dumps(now, indent=1, sort_keys=True))
        n = sum(len(v) for v in now["arms"].values())
        print(f"baseline: {len(now['bases'])} bases, {n} reference folds -> {a.out}")
        return 0
    base = json.loads(Path(a.check).read_text())
    bad, summary = [], {}
    for b, sha in base["bases"].items():
        if now["bases"][b] != sha:
            bad.append(f"base {b} framework_sha {sha} -> {now['bases'][b]}")
    for tag, rows in base["arms"].items():
        eq = 0
        for fid, r in rows.items():
            n = now["arms"][tag][fid]
            for k in ("framework_sha", "model_key"):
                if n[k] != r[k]:
                    bad.append(f"{tag} {fid} {k} {r[k]} -> {n[k]}")
            rr = _rec(tag + a.suffix, fid)
            if rr is None:
                bad.append(f"{tag}{a.suffix} {fid} missing")
                continue
            if rr.get("n_adds") or rr.get("hp_counters"):
                bad.append(f"{tag}{a.suffix} {fid} carries add fields")
            if (rr.get("trades_hash"), rr.get("n_trades")) == (r["trades_hash"], r["n_trades"]):
                eq += 1
            else:
                bad.append(f"{tag}{a.suffix} {fid} trades_hash {r['trades_hash']} -> "
                           f"{rr.get('trades_hash')}")
        summary[tag] = f"{eq}/{len(rows)}"
    print(json.dumps(summary, indent=1))
    print("\n".join(bad) if bad else "ALL EQUAL")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
