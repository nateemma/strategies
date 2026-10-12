"""Fingerprint every NNNC strategy class (feature 029, contract I4).

Scans the file stems of `NNNC/*.py` and resolves each same-named strategy class's framework
fingerprint with `evaluation.fingerprint` -- class objects only, never instantiated (an NNNC
instance loads or trains a model).

    python user_data/strategies/scripts/nnnc_fingerprints.py --out before.json
    python user_data/strategies/scripts/nnnc_fingerprints.py --out after.json --compare before.json

Values: a 16-hex sha, "LOAD_ERROR: <Exc>: <msg>", or "NOT_A_STRATEGY" (a stem whose file defines
no same-named IStrategy subclass: mixins, classifiers). --compare exits 1 on any difference.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from pathlib import Path


STRAT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STRAT))
warnings.filterwarnings("ignore")


def scan() -> dict:
    from freqtrade.strategy import IStrategy

    from evaluation import fingerprint as fp

    cfg = fp.load_config(STRAT / "config" / "config_wf.json")
    out = {}
    for py in sorted((STRAT / "NNNC").glob("*.py")):
        name = py.stem
        if not re.search(rf"^class {re.escape(name)}\b", py.read_text(errors="ignore"), re.M):
            out[name] = "NOT_A_STRATEGY"
            continue
        try:
            cls = fp.load_strategy_class(name)
            if not (isinstance(cls, type) and issubclass(cls, IStrategy)):
                out[name] = "NOT_A_STRATEGY"
                continue
            out[name], _ = fp.resolve(cls, fp.load_params_json(name), cfg)
        except Exception as exc:  # noqa: BLE001 -- recorded, compared, never hidden
            out[name] = f"LOAD_ERROR: {type(exc).__name__}: {str(exc).splitlines()[0][:160] if str(exc) else ''}"
    return out


def compare(before: dict, after: dict) -> dict:
    keys = set(before) | set(after)
    return {
        "added": sorted(k for k in keys if k not in before),
        "removed": sorted(k for k in keys if k not in after),
        "changed": sorted(k for k in keys if k in before and k in after
                          and before[k] != after[k]),
    }


def counts(d: dict) -> dict:
    return {"strategies": sum(not v.startswith(("LOAD_ERROR", "NOT_A")) for v in d.values()),
            "load_errors": sum(v.startswith("LOAD_ERROR") for v in d.values()),
            "not_a_strategy": sum(v == "NOT_A_STRATEGY" for v in d.values())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare")
    a = ap.parse_args(argv)
    res = scan()
    Path(a.out).write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    print(json.dumps(counts(res)))
    if a.compare:
        diff = compare(json.loads(Path(a.compare).read_text()), res)
        print(json.dumps(diff, indent=1))
        return 1 if any(diff.values()) else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
