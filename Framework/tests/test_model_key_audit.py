"""030 T013/T018: the model-key completeness audit and the legacy-redirect register.

A key collision (two different models, one key) is worse than a retrain, so every name that can
distinguish one NN strategy's training from another's must be classified:

* attributes -- every class attribute defined in MORE THAN ONE class across the in-scope
  families' MROs (a name defined once is the same for every strategy, so it cannot tell two
  models apart; changing such a default is a MODEL_CODE_VERSION bump) must be in exactly one of
  the merged MODEL_KEY_ATTRIBUTES / MODEL_KEY_IGNORED (or a run input);
* methods -- every method defined in more than one class must be in exactly one of the merged
  MODEL_KEY_METHODS / MODEL_KEY_METHODS_IGNORED.

freqtrade hyperopt Parameters are post-model by construction (no `.value` is read on the
training path -- research R1) and are skipped. Classes are imported, never instantiated.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
import warnings
from pathlib import Path

import pytest


STRAT = Path(__file__).resolve().parents[2]
for p in (STRAT, STRAT / "Framework"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
warnings.filterwarnings("ignore")

from Framework import ModelKey as MK  # noqa: E402


FAMILIES = ("NNNC", "NNMT", "NNPredict", "NNWavelet", "Anomaly", "Sklearn")
_GENERATED = re.compile(r"_f\d{2}(_s\d+)?(_PAP)?$")
REGISTER = STRAT / "specs" / "030-model-fingerprint-naming" / "legacy_redirects.json"


def _load(py: Path):
    d = str(py.parent)
    if d not in sys.path:
        sys.path.insert(0, d)
    name = f"_audit_{py.parent.name}_{py.stem}"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, py)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[name]


def _strategies():
    from Framework.BaseNNStrategy import BaseNNStrategy

    out = []
    for fam in FAMILIES:
        for py in sorted((STRAT / fam).glob("*.py")):
            if py.stem.startswith("test_") or _GENERATED.search(py.stem):
                continue
            if not re.search(rf"^class {re.escape(py.stem)}\b", py.read_text(errors="ignore"), re.M):
                continue
            try:
                cls = getattr(_load(py), py.stem)
            except Exception:  # noqa: BLE001 -- pre-existing load failures (Sklearn, 3 others)
                continue
            if isinstance(cls, type) and issubclass(cls, BaseNNStrategy):
                out.append(cls)
    return out


STRATEGIES = _strategies()


def _own(k) -> bool:
    return not (getattr(k, "__module__", "") or "").startswith("freqtrade") and k is not object


def _definitions():
    from freqtrade.strategy.parameters import BaseParameter

    attrs, meths = {}, {}
    for cls in STRATEGIES:
        for k in cls.__mro__:
            if not _own(k):
                continue
            for n, v in k.__dict__.items():
                if n.startswith("__") or isinstance(v, (BaseParameter, property)):
                    continue
                is_m = (callable(v) and not isinstance(v, type)) or isinstance(v, (classmethod, staticmethod))
                (meths if is_m else attrs).setdefault(n, set()).add(k.__name__)
    return attrs, meths


ATTRS, METHS = _definitions()


def test_the_scan_sees_the_families():
    assert len(STRATEGIES) > 200


@pytest.mark.parametrize("kind", ["attributes", "methods"])
def test_every_multiply_defined_name_is_classified_exactly_once(kind):
    defs = ATTRS if kind == "attributes" else METHS
    yes_name, no_name = (("MODEL_KEY_ATTRIBUTES", "MODEL_KEY_IGNORED") if kind == "attributes"
                         else ("MODEL_KEY_METHODS", "MODEL_KEY_METHODS_IGNORED"))
    unclassified, doubled = {}, {}
    for cls in STRATEGIES:
        yes, no = set(MK.merged(cls, yes_name)), set(MK.merged(cls, no_name))
        if kind == "attributes":
            yes |= set(MK.RUN_NAMES)
        for n in yes & no:
            doubled.setdefault(n, cls.__name__)
        for k in cls.__mro__:
            if not _own(k):
                continue
            for n in k.__dict__:
                if n in defs and len(defs[n]) > 1 and n not in yes and n not in no:
                    unclassified.setdefault(n, sorted(defs[n]))
    assert not doubled, f"classified both ways: {doubled}"
    assert not unclassified, "unclassified " + kind + ":\n" + "\n".join(
        f"  {n}: {', '.join(v)[:120]}" for n, v in sorted(unclassified.items()))


def _properties() -> set:
    """Names defined as properties: ModelKey.resolve reads a computed key input through its
    getter (e.g. 033's POOLED_LABEL, built from the class's own label attributes)."""
    return {n for cls in STRATEGIES for k in cls.__mro__ if _own(k)
            for n, v in k.__dict__.items() if isinstance(v, property)}


def test_every_declared_key_attribute_exists_somewhere():
    declared = set()
    for cls in STRATEGIES:
        declared |= set(MK.merged(cls, "MODEL_KEY_ATTRIBUTES"))
    props = _properties()
    missing = sorted(n for n in declared if n not in ATTRS and n not in props)
    assert not missing, f"MODEL_KEY_ATTRIBUTES names no class defines (typo?): {missing}"


def test_every_strategy_resolves_a_key():
    from evaluation.fingerprint import load_config

    cfg = load_config(STRAT / "config" / "config_wf.json")
    bad = {}
    for cls in STRATEGIES:
        try:
            MK.resolve(cls, cfg)
        except Exception as exc:  # noqa: BLE001
            bad[cls.__name__] = f"{type(exc).__name__}: {exc}"[:160]
    assert not bad, json.dumps(bad, indent=1)


def test_legacy_redirect_register_matches_the_code():
    """Every class whose model path is a class-level override is in the committed register."""
    if not REGISTER.is_file():
        pytest.skip("legacy_redirects.json not written yet (T018)")
    from Framework.BaseNNStrategy import BaseNNStrategy

    register = json.loads(REGISTER.read_text())
    now = {}
    for cls in STRATEGIES:
        definer = next(k for k in cls.__mro__ if "get_model_path" in k.__dict__)
        if definer is not BaseNNStrategy:
            now[cls.__name__] = definer.__name__
    assert now == {k: v["defined_in"] for k, v in register.items()}
