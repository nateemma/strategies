"""037 discovery (research R2): which SimpleStrategies classes the harvest covers.

In scope: every class defined in SimpleStrategies/ that subclasses SimpleStrategy (directly or via
EhlersBase / Ehlers_Indicator). Excluded with a reason: trigger arms (TriggerOnlyStrategy
subclasses -- they wrap registered triggers already scored in 021; gbb enters as the candidate
flag), the harvest wrapper AllIndicators, and modules that fail to import.

    PYTHONPATH=. ../../.venv/bin/python -m research.indicator_screen.discover [--json]
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import sys
from pathlib import Path

STRAT = Path(__file__).resolve().parents[2]
SIMPLE = STRAT / "SimpleStrategies"
NOT_STRATEGIES = {"update_inheritance"}
HARVEST_WRAPPERS = {"AllIndicators"}


def _paths() -> None:
    for p in (str(STRAT), str(SIMPLE)):
        if p not in sys.path:
            sys.path.insert(0, p)


def discover() -> dict:
    """{"classes": [{class, module, base}], "excluded": [{class, reason, detail?}]} sorted by name."""
    _paths()
    base = importlib.import_module("SimpleStrategy").SimpleStrategy
    classes, excluded = [], []
    for f in sorted(SIMPLE.glob("*.py")):
        mod = f.stem
        if mod in NOT_STRATEGIES or mod in ("SimpleStrategy", "TriggerOnly"):
            continue
        try:
            m = importlib.import_module(mod)
        except Exception as e:  # noqa: BLE001 -- any import failure is recorded, not fatal
            excluded.append({"class": mod, "reason": "import_error", "detail": repr(e)[:200]})
            continue
        for name, cls in inspect.getmembers(m, inspect.isclass):
            if cls.__module__ != mod:
                continue
            # by name: the arms import it as SimpleStrategies.TriggerOnly, a different module object
            if any(k.__name__ == "TriggerOnlyStrategy" for k in cls.__mro__):
                excluded.append({"class": name, "reason": "trigger_arm"})
            elif name in HARVEST_WRAPPERS:
                excluded.append({"class": name, "reason": "harvest_wrapper"})
            elif issubclass(cls, base) and cls is not base:
                parent = cls.__mro__[1].__name__
                classes.append({"class": name, "module": mod, "base": parent})
    classes.sort(key=lambda c: c["class"])
    excluded.sort(key=lambda c: c["class"])
    return {"classes": classes, "excluded": excluded}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    d = discover()
    if a.json:
        print(json.dumps(d, indent=1))
    else:
        print(f"in scope: {len(d['classes'])}")
        for c in d["classes"]:
            print(f"  {c['class']:<18} ({c['base']})")
        print(f"excluded: {len(d['excluded'])}")
        for c in d["excluded"]:
            print(f"  {c['class']:<18} {c['reason']} {c.get('detail', '')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
