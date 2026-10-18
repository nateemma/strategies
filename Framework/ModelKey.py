"""Model keys: locate NN weights by what shaped them, not by strategy class name (feature 030).

A model key is the canonical JSON of every input that shapes a model's trained weights:

* ``attributes`` -- the strategy's ``MODEL_KEY_ATTRIBUTES``, merged along the MRO (each class
  declares the names it owns), read from the class's DECLARED values (deep copies; never an
  instance's runtime-mutated copy, e.g. ``process_one_hot_columns`` edits ``include_list``);
* ``methods`` -- for each ``MODEL_KEY_METHODS`` name, the file and qualname of the class that
  defines it, so a subclass that overrides a label/feature/classifier method gets its own key;
* ``files`` -- pre-built scaler / PCA / GAN files by content hash (a regenerated scaler re-keys
  every model trained on it);
* ``run`` -- training timerange (``MODEL_TRAIN_TIMERANGE`` or "unpinned"), whitelist (a plot of
  whitelisted pairs keys on the configured whitelist, see ``_key_whitelist``), timeframe,
  startup candles and seed;
* ``code_version`` -- ``MODEL_CODE_VERSION``, bumped by hand when training code changes.

``MODEL_TRAINED_WITH`` pins name values the WEIGHTS were trained with that differ from the
running strategy's (e.g. a 4096-candle gate arm on weights trained with 64 startup candles). A pin
replaces the runtime value in the key, so the arm resolves to the trained model's key exactly; the
pin itself is not hashed (it would make the key differ) but is recorded where the key is reported
(the manifest's users at training/migration, the arm record's model_key).

This module is generic: it reads only the declarations; it knows no strategy family.
Lookup order (``locate``): the model index entry for the key -> the class-named legacy directory
-> the keyed location ``models/<prefix>__<hash>/``. In strict mode (``FT_MODEL_EVAL_STRICT=1``,
set by walk-forward evaluation) a missing model raises ``ModelMissing`` instead of training.
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import datetime as _dt
import enum
import hashlib
import inspect
import json
import os
from pathlib import Path
from typing import Any


SCHEMA = 1
STRAT_DIR = Path(__file__).resolve().parents[1]
INDEX_NAME = "model_index.json"
MANIFEST_NAME = "manifest.json"
LOCK_NAME = ".training.lock"
MODELS_SUBDIR = "models"
WEIGHT_SUFFIXES = (".keras", ".safetensors", ".sav", ".joblib", ".h5")
RUN_FIELDS = ("train_timerange", "whitelist", "timeframe", "startup_candle_count", "seed")
#: Names the resolver reads itself (run inputs and the declarations): classified by definition.
RUN_NAMES = ("timeframe", "startup_candle_count", "MODEL_TRAIN_TIMERANGE", "MODEL_SEED",
             "MODEL_TRAINED_WITH", "MODEL_CODE_VERSION", "MODEL_KEY_ATTRIBUTES",
             "MODEL_KEY_IGNORED", "MODEL_KEY_METHODS", "MODEL_KEY_METHODS_IGNORED")


class ModelKeyError(RuntimeError):
    """A key cannot be resolved (missing attribute, unserialisable value, bad pin)."""


class ModelMissing(ModelKeyError):
    """Strict mode: no weights exist for the resolved key."""


class ModelKeyMismatch(ModelKeyError):
    """The weights' manifest records a different key from the one resolved."""


def strict() -> bool:
    return os.environ.get("FT_MODEL_EVAL_STRICT", "") == "1"


def report_mode() -> bool:
    return os.environ.get("FT_MODEL_KEY", "") == "report"


# ------------------------------------------------------------------------------ declarations


def merged(cls, name: str) -> tuple:
    """Union of tuple attribute `name` over the MRO, base first, order kept, no duplicates."""
    out: list = []
    for k in reversed(cls.__mro__):
        for v in k.__dict__.get(name, ()) or ():
            if v not in out:
                out.append(v)
    return tuple(out)


def merged_dict(cls, name: str) -> dict:
    """Dict attribute `name` merged over the MRO; the most derived class wins per key."""
    out: dict = {}
    for k in reversed(cls.__mro__):
        out.update(k.__dict__.get(name) or {})
    return out


_MISSING = object()


def declared(cls, name: str, default=_MISSING):
    """The class-level declared value of `name` (MRO lookup), deep-copied."""
    for k in cls.__mro__:
        if name in k.__dict__:
            return copy.deepcopy(k.__dict__[name])
    if default is not _MISSING:
        return default
    raise ModelKeyError(f"{cls.__name__}: model-key attribute {name!r} is not declared")


# ---------------------------------------------------------------------------- serialisation


def jsonable(v: Any, where: str = "") -> Any:
    if v is None or isinstance(v, (bool, int, str)):
        return v
    if isinstance(v, enum.Enum):
        return f"{type(v).__name__}.{v.name}"
    if isinstance(v, float):
        return v
    if isinstance(v, Path):
        return str(v)
    if dataclasses.is_dataclass(v) and not isinstance(v, type):
        return {f.name: jsonable(getattr(v, f.name), f"{where}.{f.name}")
                for f in dataclasses.fields(v)}
    if isinstance(v, dict):
        return {str(k): jsonable(x, f"{where}[{k}]") for k, x in sorted(v.items(), key=lambda i: str(i[0]))}
    if isinstance(v, (list, tuple)):
        return [jsonable(x, f"{where}[]") for x in v]
    if isinstance(v, (set, frozenset)):
        return sorted(jsonable(x, f"{where}{{}}") for x in v)
    if isinstance(v, type):
        return _where(v)
    raise ModelKeyError(f"model-key input {where!r} has unserialisable type {type(v).__name__}")


def canonical(key: dict) -> str:
    return json.dumps(key, sort_keys=True, separators=(",", ":"))


def key_hash(key: dict) -> str:
    return hashlib.sha256(canonical(key).encode()).hexdigest()[:16]


def _safe(s: str) -> str:
    return "".join(c if c.isalnum() or c in ".-" else "" for c in str(s).lower())


def key_prefix(key: dict) -> str:
    a = key["attributes"]
    ct = str(a.get("classifier_type") or "model").split(".")[-1]
    lbl = a.get("TRAINING_TYPE")
    if isinstance(lbl, int):  # TRAINING_TYPE is declared as a plain int; name it for reading
        from Framework.TrainingSignals import LabelMethod

        try:
            lbl = LabelMethod(lbl).name
        except ValueError:
            pass
    lbl = str(lbl or "label").split(".")[-1]
    return (f"{_safe(ct)}_{_safe(lbl)}_g{_safe(a.get('MIN_BUY_GAIN_THRESHOLD'))}"
            f"_h{_safe(a.get('HORIZON'))}")


def _where(k) -> str:
    """File (relative to the strategies dir) + qualname: stable across module loaders."""
    try:
        f = Path(inspect.getfile(k)).resolve()
        f = f.relative_to(STRAT_DIR) if f.is_relative_to(STRAT_DIR) else Path(k.__module__)
    except (TypeError, OSError):
        f = Path(k.__module__)
    return f"{f.as_posix()}:{k.__qualname__}"


# ------------------------------------------------------------------------------- resolution

_FILE_CACHE: dict = {}


def file_sha(path: Path) -> str:
    """sha256[:16] of a file, or "absent"; cached by (path, mtime, size)."""
    p = Path(path)
    if not p.is_file():
        return "absent"
    st = p.stat()
    ck = (str(p), st.st_mtime_ns, st.st_size)
    if ck not in _FILE_CACHE:
        h = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        _FILE_CACHE[ck] = h.hexdigest()[:16]
    return _FILE_CACHE[ck]


def dir_sha(path: Path) -> str:
    p = Path(path)
    if not p.is_dir():
        return "absent"
    parts = [f"{q.relative_to(p).as_posix()}={file_sha(q)}" for q in sorted(p.rglob("*")) if q.is_file()]
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16] if parts else "absent"


def _bare(cls):
    """An uninitialised instance: enough for methods that read only class state (no
    freqtrade __init__, so nothing loads)."""
    return object.__new__(cls)


def storage_of(cls) -> Path:
    """The class's storage root (an override may call super(), so call it on an instance)."""
    return Path(_bare(cls).get_storage_location())


def _file_inputs(cls, attrs: dict, storage: Path) -> dict:
    files = {"main_scaler": file_sha(storage / f"{declared(cls, 'main_scaler_name')}.pkl")}
    if declared(cls, "use_pca_reduction", False):
        files["pca"] = file_sha(storage / f"{declared(cls, 'pca_name')}.pkl")
    if declared(cls, "use_post_gan_scaling", False):
        files["tensor_scaler"] = file_sha(storage / "main_tensor_scaler.pkl")
    gan = declared(cls, "gan_type", None)
    if gan is not None and getattr(gan, "name", "NONE") != "NONE" and declared(cls, "gan_augment", True):
        from GANs.paths import gan_save_path

        files["gan_generator"] = dir_sha(Path(gan_save_path(
            str(storage), gan, use_pca=bool(declared(cls, "use_pca_reduction", False)),
            post_gan_scaling=bool(declared(cls, "use_post_gan_scaling", False)))))
    return files


def _key_whitelist(config: dict):
    """The whitelist the key uses: the run's, except in plot mode. freqtrade's `-p` overwrites
    exchange.pair_whitelist with the plotted pairs, which would key a model nobody trained; when
    every plotted pair is in the configured whitelist (kept in `original_config`), a plot keys on
    that whitelist. Every other run mode keys on the run's whitelist, unchanged."""
    wl = (config.get("exchange") or {}).get("pair_whitelist")
    mode = config.get("runmode")
    if getattr(mode, "value", mode) == "plot":
        orig = ((config.get("original_config") or {}).get("exchange") or {}).get("pair_whitelist")
        if orig and wl and set(wl) <= set(orig):
            return orig
    return wl


def resolve(cls, config: dict, storage: Path | None = None) -> dict:
    """The model key of strategy class `cls` under freqtrade `config`. Never instantiates.
    `storage` overrides the class's storage root (an instance-level override)."""
    names = merged(cls, "MODEL_KEY_ATTRIBUTES")
    pins = merged_dict(cls, "MODEL_TRAINED_WITH")
    unknown = [p for p in pins if p not in names and p not in ("startup_candle_count", "timeframe")]
    if unknown:
        raise ModelKeyError(f"{cls.__name__}: MODEL_TRAINED_WITH names non-key inputs {unknown}")
    attrs = {}
    for n in names:
        # None when this class's MRO never defines `n` (a name another mixin owns, or one read
        # by getattr with a default); the audit test checks every listed name exists somewhere.
        v = pins[n] if n in pins else declared(cls, n, None)
        if isinstance(v, property):  # a computed name (e.g. NNWavelet's coeff_scaler_name)
            v = v.fget(_bare(cls))
        attrs[n] = jsonable(v, n)
    methods = {}
    for m in merged(cls, "MODEL_KEY_METHODS"):
        definer = next((k for k in cls.__mro__ if m in k.__dict__), None)
        methods[m] = None if definer is None else _where(definer)
    seed = declared(cls, "MODEL_SEED", None)
    if seed is None:
        seed = int(os.environ.get("MLX_SEED", "42"))
    wl = _key_whitelist(config)
    if not wl:
        raise ModelKeyError("config has no exchange.pair_whitelist; the model key needs it")
    run = {
        "train_timerange": declared(cls, "MODEL_TRAIN_TIMERANGE", None) or "unpinned",
        "whitelist": list(wl),
        "timeframe": pins.get("timeframe", config.get("timeframe") or declared(cls, "timeframe")),
        "startup_candle_count": int(pins.get("startup_candle_count",
                                             declared(cls, "startup_candle_count"))),
        "seed": int(seed),
    }
    storage = Path(storage) if storage is not None else storage_of(cls)
    key = {"version": SCHEMA, "code_version": declared(cls, "MODEL_CODE_VERSION", 1),
           "attributes": attrs, "methods": methods, "files": _file_inputs(cls, attrs, storage),
           "models": {}, "run": run}
    if declared(cls, "gan_entropy_guidance", False):
        key["models"]["entropy_guidance"] = f"legacy:{declared(cls, 'gan_entropy_guidance_model')}"
    return key


# ------------------------------------------------------------------------------ index & files


def index_path(storage: Path) -> Path:
    return Path(storage) / INDEX_NAME


def load_index(storage: Path) -> dict:
    p = index_path(storage)
    if not p.is_file():
        return {"version": SCHEMA, "entries": {}, "conflicts": []}
    return json.loads(p.read_text())


def save_index(storage: Path, idx: dict) -> None:
    p = index_path(storage)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(idx, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, p)


def weights_exist(stem: Path) -> bool:
    return any(Path(f"{stem}{s}").is_file() for s in WEIGHT_SUFFIXES)


@dataclasses.dataclass
class Location:
    path: str            # the ".keras" path handed to the classifier (it rewrites the suffix)
    kind: str            # keyed | migrated | legacy | absent
    hash: str            # the key hash, or "legacy:<Class>"
    prefix: str
    key: dict


def _legacy_stem(storage: Path, name: str) -> Path:
    return Path(storage) / name / name


def locate_for(cls, config: dict, class_name: str | None = None,
               storage: Path | None = None) -> Location:
    """Where `cls`'s weights are (see module docstring). Never trains; strict mode raises."""
    name = class_name or cls.__name__
    storage = Path(storage) if storage is not None else storage_of(cls)
    key = resolve(cls, config, storage)
    h, prefix = key_hash(key), key_prefix(key)
    idx = load_index(storage)
    conflicted = {c["hash"] for c in idx.get("conflicts", [])}
    ent = idx["entries"].get(h)
    if ent and h not in conflicted:
        stem = Path(storage) / ent["stem"]
        if weights_exist(stem):
            return Location(f"{stem}.keras", ent["kind"], h, prefix, key)
    leg = _legacy_stem(storage, name)
    if weights_exist(leg):
        return Location(f"{leg}.keras", "legacy", f"legacy:{name}", prefix, key)
    new = Path(storage) / MODELS_SUBDIR / f"{prefix}__{h}" / f"{prefix}__{h}"
    if weights_exist(new):
        return Location(f"{new}.keras", "keyed", h, prefix, key)
    if strict():
        raise ModelMissing(
            f"{name}: no weights for model key {prefix}__{h} (strict walk-forward evaluation "
            f"refuses to train). Key: {canonical(key)}")
    return Location(f"{new}.keras", "absent", h, prefix, key)


def manifest_path(location: Location) -> Path:
    return Path(location.path).parent / MANIFEST_NAME


def diff_keys(a: dict, b: dict, path: str = "") -> list[str]:
    out = []
    for k in sorted(set(a) | set(b)):
        p = f"{path}.{k}" if path else k
        if k not in a or k not in b:
            out.append(p)
        elif isinstance(a[k], dict) and isinstance(b[k], dict):
            out += diff_keys(a[k], b[k], p)
        elif a[k] != b[k]:
            out.append(p)
    return out


def check_manifest(location: Location) -> None:
    """Halt when the weights' manifest records a different key (FR-008). Legacy is exempt."""
    if location.kind not in ("keyed", "migrated"):
        return
    mp = manifest_path(location)
    if not mp.is_file():
        raise ModelKeyMismatch(f"{location.path}: indexed as {location.kind} but has no manifest")
    man = json.loads(mp.read_text())
    if man.get("hash") != location.hash or man.get("key") != location.key:
        raise ModelKeyMismatch(
            f"{location.path}: manifest key differs from the resolved key in "
            f"{diff_keys(man.get('key') or {}, location.key)}")


def write_manifest(directory: Path, key: dict, kind: str, provenance: dict, weights: dict,
                   users: list) -> Path:
    h = key_hash(key)
    man = {"key": key, "hash": h, "prefix": key_prefix(key), "kind": kind,
           "provenance": provenance, "weights": weights, "users": users,
           "written": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")}
    p = Path(directory) / MANIFEST_NAME
    p.write_text(json.dumps(man, indent=1, sort_keys=True) + "\n")
    return p


@contextlib.contextmanager
def training_lock(location: Location):
    """Exclusive lock while training into a key (R12): a second trainer fails loudly."""
    d = Path(location.path).parent
    d.mkdir(parents=True, exist_ok=True)
    lock = d / LOCK_NAME
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise ModelKeyError(f"{d}: another run is training this model ({lock.read_text().strip()})")
    with os.fdopen(fd, "w") as f:
        f.write(f"pid {os.getpid()} {_dt.datetime.now(_dt.UTC).isoformat(timespec='seconds')}\n")
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def record_training(location: Location, class_name: str, timerange: str | None) -> None:
    """After a successful train into a keyed location: manifest + index entry (FR-010)."""
    if location.kind == "legacy":
        return
    stem = Path(location.path).with_suffix("")
    storage = Path(location.path).parents[2]
    files = [Path(f"{stem}{s}") for s in WEIGHT_SUFFIXES if Path(f"{stem}{s}").is_file()]
    write_manifest(stem.parent, location.key, "keyed",
                   {"kind": "trained", "timerange": timerange, "class": class_name},
                   {f.name: file_sha(f) for f in files}, [class_name])
    idx = load_index(storage)
    idx["entries"][location.hash] = {"stem": str(stem.relative_to(storage)), "kind": "keyed"}
    save_index(storage, idx)


def emit_sink(class_name: str, location: Location) -> None:
    """Hand the resolved key to regime/run_arm.py's record (R12) when it asks for it."""
    d = os.environ.get("FT_MODEL_KEY_SINK")
    if not d:
        return
    Path(d).mkdir(parents=True, exist_ok=True)
    (Path(d) / f"{class_name}_modelkey.json").write_text(json.dumps(
        {"class": class_name, "hash": location.hash, "prefix": location.prefix,
         "kind": location.kind, "path": location.path}) + "\n")
