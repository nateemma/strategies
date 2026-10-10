"""Canonical dump of a strategy's class-level configuration (feature 023 T032, SC-008).

Proves a change did not move a strategy it should not touch -- the live dry-run's
`MomentumRegimeBasket15mFast`, which does not inherit BaseStrategy, so the framework fingerprint's
preset members barely apply to it. Instead this dumps EVERY public, non-callable class attribute
across the MRO (repr for values JSON cannot hold) plus its params JSON, after the merge freqtrade
applies. Run it once per tree, each in its own process (`--strat-dir` points at a git worktree
for the "before" side), and diff the outputs.

Never instantiates the strategy.
"""
from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import sys
from pathlib import Path


def _find(strat_dir: Path, name: str) -> Path:
    hits = [p for p in strat_dir.glob(f"*/{name}.py")]
    if len(hits) != 1:
        raise SystemExit(f"expected one {name}.py under {strat_dir}, found {hits}")
    return hits[0]


def _jsonable(v):
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in sorted(v.items(), key=lambda kv: str(kv[0]))}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    val = getattr(v, "value", None)               # hyperopt Parameter -> its default value
    if val is not None and hasattr(v, "space"):
        return {"parameter": type(v).__name__, "default": _jsonable(getattr(v, "default", val))}
    return repr(v)


def dump(strat_dir: Path, name: str) -> dict:
    path = _find(strat_dir, name)
    for p in (str(strat_dir), str(strat_dir / "utils"), str(path.parent)):
        if p not in sys.path:
            sys.path.insert(0, p)
    spec = importlib.util.spec_from_file_location(f"_dump_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cls = getattr(mod, name)
    attrs = {}
    for k in sorted(dir(cls)):
        if k.startswith("_"):
            continue
        v = inspect.getattr_static(cls, k)
        if callable(v) and not hasattr(v, "space") or isinstance(v, (staticmethod, classmethod, property)):
            continue
        attrs[k] = _jsonable(getattr(cls, k))
    js = path.with_suffix(".json")
    return {"strategy": name, "mro": [c.__name__ for c in cls.__mro__], "attributes": attrs,
            "params_json": json.loads(js.read_text()) if js.is_file() else None}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--strategy", required=True)
    ap.add_argument("--strat-dir", default=str(Path(__file__).resolve().parents[1]))
    a = ap.parse_args(argv)
    print(json.dumps(dump(Path(a.strat_dir).resolve(), a.strategy), indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
