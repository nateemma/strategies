"""030 T002: which class decides each NN strategy's model path?

For every class in the in-scope family directories that subclasses Framework's BaseNNStrategy,
records the class whose `__dict__` defines its effective `get_model_path` (via the MRO), and --
when that is not BaseNNStrategy -- the redirect target parsed from the override's source.
Classes are imported, never instantiated.

    ../../.venv/bin/python specs/030-model-fingerprint-naming/scan_overrides.py --out OUT.json
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import re
import sys
import warnings
from pathlib import Path


STRAT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(STRAT))
sys.path.insert(0, str(STRAT / "Framework"))
warnings.filterwarnings("ignore")

FAMILIES = ("NNNC", "NNMT", "NNPredict", "NNWavelet", "Anomaly", "Sklearn")
_FOLD = re.compile(r"_f\d{2}(_s\d+)?(_PAP)?$")


def _load(py: Path):
    d = str(py.parent)
    if d not in sys.path:
        sys.path.insert(0, d)
    name = f"_scan_{py.parent.name}_{py.stem}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, py)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _target(cls) -> str | None:
    """Best-effort literal redirect target of an override (documentation only)."""
    try:
        src = inspect.getsource(cls.__dict__["get_model_path"])
    except (OSError, TypeError, KeyError):
        return None
    m = re.search(r'"([A-Za-z0-9_]+)/"', src) or re.search(r"MODEL_SOURCE|reuse_model_from", src)
    return m.group(1) if m and m.groups() else (m.group(0) if m else "computed")


def scan(include_generated: bool = False) -> dict:
    from Framework.BaseNNStrategy import BaseNNStrategy

    out = {}
    for fam in FAMILIES:
        for py in sorted((STRAT / fam).glob("*.py")):
            if py.stem.startswith("test_") or (_FOLD.search(py.stem) and not include_generated):
                continue
            if not re.search(rf"^class {re.escape(py.stem)}\b", py.read_text(errors="ignore"), re.M):
                continue
            try:
                cls = getattr(_load(py), py.stem)
            except Exception as exc:  # noqa: BLE001 -- recorded, not fatal
                out[py.stem] = {"family": fam, "error": f"{type(exc).__name__}: {exc}"[:200]}
                continue
            if not (isinstance(cls, type) and issubclass(cls, BaseNNStrategy)):
                continue
            definer = next(k for k in cls.__mro__ if "get_model_path" in k.__dict__)
            row = {"family": fam, "defined_in": definer.__name__}
            if definer is not BaseNNStrategy:
                row["target"] = _target(definer)
            out[py.stem] = row
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res = scan()
    Path(a.out).write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    by = {}
    for r in res.values():
        by[r.get("defined_in", "ERROR")] = by.get(r.get("defined_in", "ERROR"), 0) + 1
    print(f"{len(res)} classes; by path definer: {dict(sorted(by.items(), key=lambda x: -x[1]))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
