"""Model-key index tool (feature 030, contracts/resolver.md).

    key <Class> [--plan P --fold fNN] [--seed N] [--config C]
                                  prefix, hash, canonical key, location, weights present (0/1)
    path <Class> [--config C]     the weight path a run of <Class> would use (test_strat.sh -f)
    migrate [--dry-run]           index the walk-forward sets (keys rebuilt from their plans) and
                                  every single-model directory (per-class legacy keys); manifests
                                  beside the weights; weights never moved. exit 3 on a conflict
    audit                         the completeness audit + legacy-redirect register (pytest)
    verify-bytes --write <json>   snapshot {relpath: {size, sha256}} of every weight/artefact file
                                  under saved_data/ (manifests and the index excluded)
    verify-bytes <json>           compare against a snapshot; exit 1 on any difference

Run with the repo venv from user_data/strategies:

    ../../.venv/bin/python scripts/model_index.py verify-bytes --write snap.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


STRAT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STRAT))
SAVED = STRAT / "saved_data"
# Written by 030 itself; never weights, so never part of the no-change check.
EXCLUDED_NAMES = {"model_index.json", "manifest.json", ".training.lock"}


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot(root: Path = SAVED) -> dict:
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_symlink() or not p.is_file() or p.name in EXCLUDED_NAMES:
            continue
        if any((root / part).is_symlink() for part in p.relative_to(root).parents
               if part != Path(".")):
            continue
        out[str(p.relative_to(root))] = {"size": p.stat().st_size, "sha256": _sha256(p)}
    return out


def compare(before: dict, after: dict) -> list[str]:
    diffs = [f"removed: {k}" for k in sorted(set(before) - set(after))]
    diffs += [f"added: {k}" for k in sorted(set(after) - set(before))]
    diffs += [f"changed: {k}" for k in sorted(set(before) & set(after)) if before[k] != after[k]]
    return diffs


def cmd_verify_bytes(a) -> int:
    snap = snapshot()
    if a.write:
        Path(a.json).write_text(json.dumps(snap, indent=1, sort_keys=True) + "\n")
        print(f"wrote {len(snap)} files -> {a.json}")
        return 0
    diffs = compare(json.loads(Path(a.json).read_text()), snap)
    for d in diffs:
        print(d)
    print(f"{len(snap)} files, {len(diffs)} differences")
    return 1 if diffs else 0


# ------------------------------------------------------------------------------ key / path

DEFAULT_CONFIG = STRAT / "config" / "config.json"
#: walk-forward sets: (plan, seeds). WF's seed-7 folds were run for f00/f04/f08/f12 only.
WF_PLANS = [("plan_nnnc.json", (1, 7)), ("plan_NNNC_A_Raw_Open.json", (1,)),
            ("plan_NNNC_ForwardMAE_Raw.json", (1,)), ("plan_NNNC_GbbRaw_Open.json", (1,)),
            ("plan_NNNC_GbbFwdMae_Open.json", (1,))]
NOT_MODELS = {"models", "GANs", "GANs_PCA", "GANs_PostScale", "Discriminators",
              "labelfree_warmstart", "_ablate"}


def _config(path) -> dict:
    from evaluation.fingerprint import load_config

    p = Path(path)
    if not p.is_absolute():  # plans carry repo-root-relative paths; CLI users pass either
        p = next((c for c in (STRAT.parents[1] / p, STRAT / p, Path.cwd() / p) if c.is_file()), p)
    return load_config(p)


def _plan(name: str) -> dict:
    p = STRAT / "evaluation" / "folds" / (name if name.endswith(".json") else f"plan_{name}.json")
    return json.loads(p.read_text())


def fold_class(base, fold: dict, seed: int):
    """The class a walk-forward fold runs: `base` pinned to the fold's training window/seed
    (what evaluation/run_folds.py's TEMPLATE writes since 030)."""
    name = fold["strategy_class"] + (f"_s{seed}" if seed != 1 else "")
    return type(name, (base,), {"MODEL_TRAIN_TIMERANGE": fold["train_timerange"],
                                "MODEL_SEED": seed, "__module__": base.__module__})


def _resolve_cls(a):
    from evaluation.fingerprint import load_strategy_class

    cls = load_strategy_class(a.cls)
    if a.plan:
        plan = _plan(a.plan)
        f = next(x for x in plan["folds"] if x["fold_id"] == a.fold)
        return fold_class(cls, f, a.seed), _config(plan["config"])
    return cls, _config(a.config)


def cmd_key(a) -> int:
    from Framework import ModelKey as MK

    from Framework.BaseNNStrategy import BaseNNStrategy

    cls, cfg = _resolve_cls(a)
    definer = next(k for k in cls.__mro__ if "get_model_path" in k.__dict__)
    if definer is not BaseNNStrategy:  # a recorded legacy redirect decides the path itself
        path = definer.get_model_path(object.__new__(cls))
        print(f"class   {cls.__name__}\nkind    legacy-redirect ({definer.__name__})\npath    {path}")
        return 0 if MK.weights_exist(Path(path).with_suffix("")) else 1
    loc = MK.locate_for(cls, cfg, cls.__name__)
    print(f"class   {cls.__name__}\nprefix  {loc.prefix}\nhash    {MK.key_hash(loc.key)}\n"
          f"kind    {loc.kind}\npath    {loc.path}")
    print("key     " + MK.canonical(loc.key))
    return 0 if loc.kind != "absent" else 1


def cmd_path(a) -> int:
    from evaluation.fingerprint import load_strategy_class
    from Framework import ModelKey as MK
    from Framework.BaseNNStrategy import BaseNNStrategy

    cls = load_strategy_class(a.cls)
    definer = next(k for k in cls.__mro__ if "get_model_path" in k.__dict__)
    if definer is not BaseNNStrategy:  # a legacy redirect: ask the override itself
        print(definer.get_model_path(object.__new__(cls)))
    else:
        print(MK.locate_for(cls, _config(a.config), cls.__name__).path)
    return 0


# ---------------------------------------------------------------------------------- migrate


def _weight_files(stem: Path) -> list[Path]:
    from Framework import ModelKey as MK

    return [Path(f"{stem}{s}") for s in MK.WEIGHT_SUFFIXES if Path(f"{stem}{s}").is_file()]


def migrate(dry: bool) -> int:
    from evaluation.fingerprint import load_strategy_class
    from Framework import ModelKey as MK

    idx = {"version": MK.SCHEMA, "entries": {}, "conflicts": []}
    rows, seen_dirs, weights_by_hash, manifests = [], set(), {}, []
    for plan_name, seeds in WF_PLANS:
        plan = _plan(plan_name)
        base = load_strategy_class(plan["base_strategy"])
        cfg = _config(plan["config"])
        for seed in seeds:
            for f in plan["folds"]:
                d = Path(f["model_dir"]).name + (f"_s{seed}" if seed != 1 else "")
                stem = SAVED / d / d
                files = _weight_files(stem)
                if not files:
                    if seed == 1:
                        rows.append(("MISSING", d, plan_name, "-"))
                    continue
                cls = fold_class(base, f, seed)
                key = MK.resolve(cls, cfg)
                h = MK.key_hash(key)
                wsha = {p.name: _sha256(p) for p in files}
                seen_dirs.add(d)
                if h in weights_by_hash and weights_by_hash[h][1] != sorted(wsha.values()):
                    idx["conflicts"].append({"hash": h, "paths": [weights_by_hash[h][0], d],
                                             "reason": "one key, different weight bytes"})
                    rows.append(("CONFLICT", d, plan_name, h))
                    continue
                weights_by_hash[h] = (d, sorted(wsha.values()))
                idx["entries"][h] = {"stem": f"{d}/{d}", "kind": "migrated"}
                manifests.append((SAVED / d, key, "migrated",
                                  {"kind": "reconstructed-from-plan", "plan": plan_name,
                                   "run_id": plan.get("run_id"), "params_sha": plan.get("params_sha"),
                                   "fold": f["fold_id"], "seed": seed,
                                   "train_timerange": f["train_timerange"]},
                                  wsha, [cls.__name__]))
                rows.append(("migrated", d, plan_name, h))
    for d in sorted(p for p in SAVED.iterdir() if p.is_dir() and not p.is_symlink()):
        if d.name in NOT_MODELS or d.name in seen_dirs:
            continue
        files = _weight_files(d / d.name)
        if not files:
            rows.append(("EMPTY", d.name, "-", "-"))
            continue
        lk = f"legacy:{d.name}"
        idx["entries"][lk] = {"stem": f"{d.name}/{d.name}", "kind": "legacy"}
        manifests.append((d, {"legacy_class": d.name, "unknown": [
            "attributes", "methods", "files", "run.train_timerange", "run.seed", "run.whitelist"]},
            "legacy", {"kind": "legacy-unknown"},
            {p.name: _sha256(p) for p in files}, [d.name]))
        rows.append(("legacy", d.name, "-", lk))
    conflicted = {c["hash"] for c in idx["conflicts"]}
    for c in conflicted:
        idx["entries"].pop(c, None)
    counts = {}
    for r in rows:
        counts[r[0]] = counts.get(r[0], 0) + 1
    for r in rows:
        if r[0] != "migrated" and r[0] != "legacy":
            print("  ".join(map(str, r)))
    print(f"counts: {counts}")
    if dry:
        print("dry run: nothing written")
        return 3 if conflicted else 0
    for directory, key, kind, prov, wsha, users in manifests:
        if kind == "legacy":
            man = {"key": key, "hash": f"legacy:{directory.name}", "prefix": None, "kind": kind,
                   "provenance": prov, "weights": wsha, "users": users}
            (directory / MK.MANIFEST_NAME).write_text(json.dumps(man, indent=1, sort_keys=True) + "\n")
        else:
            if MK.key_hash(key) in conflicted:
                continue
            MK.write_manifest(directory, key, kind, prov, wsha, users)
            link = SAVED / MK.MODELS_SUBDIR / f"{MK.key_prefix(key)}__{MK.key_hash(key)}"
            link.parent.mkdir(exist_ok=True)
            if link.is_symlink():
                link.unlink()
            link.symlink_to(Path("..") / directory.name)
    MK.save_index(SAVED, idx)
    print(f"index: {len(idx['entries'])} entries, {len(idx['conflicts'])} conflicts -> "
          f"{MK.index_path(SAVED)}")
    return 3 if conflicted else 0


def cmd_migrate(a) -> int:
    return migrate(a.dry_run)


def cmd_audit(a) -> int:
    import subprocess

    return subprocess.call([sys.executable, "-m", "pytest", "-q",
                            str(STRAT / "Framework" / "tests" / "test_model_key_audit.py")])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    k = sub.add_parser("key")
    k.add_argument("cls")
    k.add_argument("--plan")
    k.add_argument("--fold")
    k.add_argument("--seed", type=int, default=1)
    k.add_argument("--config", default=str(DEFAULT_CONFIG))
    k.set_defaults(fn=cmd_key)
    pth = sub.add_parser("path")
    pth.add_argument("cls")
    pth.add_argument("--config", default=str(DEFAULT_CONFIG))
    pth.set_defaults(fn=cmd_path)
    m = sub.add_parser("migrate")
    m.add_argument("--dry-run", action="store_true")
    m.set_defaults(fn=cmd_migrate)
    au = sub.add_parser("audit")
    au.set_defaults(fn=cmd_audit)
    v = sub.add_parser("verify-bytes")
    v.add_argument("json")
    v.add_argument("--write", action="store_true")
    v.set_defaults(fn=cmd_verify_bytes)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
