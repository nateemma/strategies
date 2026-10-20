"""030 T009/T015: model keys -- deterministic, sensitive to every weight-affecting input,
blind to post-model settings, keyed on file CONTENT; manifest, index, strict mode and lock.

Stub strategies subclass BaseNNStrategy with a tmp storage root; nothing is instantiated, no
model is built or trained.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


STRAT = Path(__file__).resolve().parents[2]
if str(STRAT) not in sys.path:
    sys.path.insert(0, str(STRAT))

from Framework import ModelKey as MK  # noqa: E402
from Framework.BaseNNStrategy import BaseNNStrategy  # noqa: E402


CONFIG = {"exchange": {"pair_whitelist": ["SOL/USDT", "LINK/USDT"]}, "timeframe": "15m"}


@pytest.fixture
def storage(tmp_path, monkeypatch):
    (tmp_path / "main_scaler.pkl").write_bytes(b"scaler-v1")
    monkeypatch.setattr(BaseNNStrategy, "get_storage_location", lambda self: str(tmp_path) + "/")
    monkeypatch.delenv("FT_MODEL_EVAL_STRICT", raising=False)
    monkeypatch.delenv("MLX_SEED", raising=False)
    return tmp_path


def _cls(name="Stub", base=BaseNNStrategy, **attrs):
    return type(name, (base,), dict(attrs))


def _h(cls, config=CONFIG):
    return MK.key_hash(MK.resolve(cls, config))


# ------------------------------------------------------------------------------ determinism


def test_key_is_deterministic_and_ignores_the_class_name(storage):
    a, b = _cls("A"), _cls("TotallyDifferentName")
    assert _h(a) == _h(b) == _h(a)
    assert len(_h(a)) == 16


def test_key_is_identical_in_a_fresh_process(storage):
    code = (
        "import sys;sys.path.insert(0,%r);"
        "from Framework import ModelKey as MK;from Framework.BaseNNStrategy import BaseNNStrategy as B;"
        "B.get_storage_location=lambda self:%r;"
        "print(MK.key_hash(MK.resolve(type('X',(B,),{}),%r)))"
    ) % (str(STRAT), str(storage) + "/", CONFIG)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip().splitlines()[-1] == _h(_cls())


def test_prefix_reads_classifier_label_gain_horizon(storage):
    p = MK.key_prefix(MK.resolve(_cls(), CONFIG))
    assert p.endswith(f"_g{BaseNNStrategy.MIN_BUY_GAIN_THRESHOLD}_h{BaseNNStrategy.HORIZON}")


# ------------------------------------------------------------------------------ sensitivity


def _bump(v):
    if isinstance(v, bool):
        return not v
    if isinstance(v, (int, float)):
        return v + 1
    if isinstance(v, str):
        return v + "_x"
    if v is None:
        return 0.123
    if isinstance(v, (list, tuple)):
        return list(v) + ["__extra__"]
    if isinstance(v, dict):
        return {**v, "__extra__": 1}
    return None


def _attr_names():
    return MK.merged(BaseNNStrategy, "MODEL_KEY_ATTRIBUTES")


@pytest.mark.parametrize("name", _attr_names())
def test_every_declared_attribute_moves_the_key(storage, name):
    base = _cls()
    v = MK.declared(base, name, None)  # None: owned by a mixin this stub lacks
    new = _bump(v)
    if new is None:
        pytest.skip(f"{name}: {type(v).__name__} has no generic bump (covered by its own test)")
    assert _h(_cls(**{name: new})) != _h(base), name


@pytest.mark.parametrize("name", MK.merged(BaseNNStrategy, "MODEL_KEY_IGNORED"))
def test_every_ignored_attribute_leaves_the_key(storage, name):
    if name == "_abc_impl":
        pytest.skip("ABC machinery; cannot be overridden on a subclass")
    base = _cls()
    v = MK.declared(base, name, None)
    assert _h(_cls(**{name: _bump(v) if _bump(v) is not None else object()})) == _h(base), name


def test_training_path_override_moves_the_key_inference_override_does_not(storage):
    base = _cls()
    lab = _cls(get_training_labels=lambda self, df: None)
    ent = _cls(get_entry_conditions=lambda self, df: None)
    assert _h(lab) != _h(base)
    assert _h(ent) == _h(base)


def test_scaler_content_moves_the_key(storage):
    before = _h(_cls())
    (storage / "main_scaler.pkl").write_bytes(b"scaler-v2-regenerated")
    assert _h(_cls()) != before


@pytest.mark.parametrize("attrs", [
    {"MODEL_TRAIN_TIMERANGE": "20210101-20230101"}, {"MODEL_SEED": 7},
    {"startup_candle_count": 4096}, {"MODEL_CODE_VERSION": 2},
])
def test_run_inputs_move_the_key(storage, attrs):
    assert _h(_cls(**attrs)) != _h(_cls())


def test_whitelist_and_its_order_move_the_key(storage):
    flipped = {**CONFIG, "exchange": {"pair_whitelist": ["LINK/USDT", "SOL/USDT"]}}
    assert _h(_cls(), flipped) != _h(_cls())


def _with_pairs(pairs, runmode):
    """A config as freqtrade builds it after `-p`: pair_whitelist overwritten, original_config kept."""
    return {**CONFIG, "runmode": runmode, "exchange": {"pair_whitelist": list(pairs)},
            "original_config": CONFIG}


def test_plot_of_a_whitelisted_pair_keys_on_the_configured_whitelist(storage):
    from freqtrade.enums import RunMode

    assert _h(_cls(), _with_pairs(["SOL/USDT"], RunMode.PLOT)) == _h(_cls())


def test_pair_override_still_moves_the_key_outside_plot(storage):
    from freqtrade.enums import RunMode

    for mode in (RunMode.BACKTEST, RunMode.DRY_RUN, RunMode.LIVE, RunMode.HYPEROPT):
        assert _h(_cls(), _with_pairs(["SOL/USDT"], mode)) != _h(_cls())


def test_plot_of_a_pair_outside_the_whitelist_is_not_redirected(storage):
    from freqtrade.enums import RunMode

    assert _h(_cls(), _with_pairs(["BTC/USDT"], RunMode.PLOT)) != _h(_cls())


def test_seed_defaults_to_mlx_seed_then_42(storage, monkeypatch):
    assert MK.resolve(_cls(), CONFIG)["run"]["seed"] == 42
    monkeypatch.setenv("MLX_SEED", "7")
    assert MK.resolve(_cls(), CONFIG)["run"]["seed"] == 7
    assert MK.resolve(_cls(MODEL_SEED=1), CONFIG)["run"]["seed"] == 1


def test_trained_with_pin_reuses_the_trained_key(storage):
    trained = _cls(startup_candle_count=64)
    gate_arm = _cls(startup_candle_count=4096, MODEL_TRAINED_WITH={"startup_candle_count": 64})
    assert MK.resolve(gate_arm, CONFIG) == MK.resolve(trained, CONFIG)
    assert MK.merged_dict(gate_arm, "MODEL_TRAINED_WITH") == {"startup_candle_count": 64}
    assert _h(_cls(startup_candle_count=4096)) != _h(trained)  # without the pin: a new model


def test_pin_on_a_non_key_input_is_refused(storage):
    with pytest.raises(MK.ModelKeyError, match="non-key"):
        MK.resolve(_cls(MODEL_TRAINED_WITH={"stoploss": -0.1}), CONFIG)


def test_instance_mutation_of_a_list_does_not_move_the_key(storage):
    cls = _cls(include_list=["a", "b"])
    before = _h(cls)
    inst = object.__new__(cls)
    inst.include_list.append("mutated")  # what process_one_hot_columns does, in place
    cls.include_list = ["a", "b"]         # the declared value is what the key reads
    assert _h(cls) == before


# ------------------------------------------------------------------- index, manifest, strict


def _train_stub(storage, cls, name="Stub"):
    loc = MK.locate_for(cls, CONFIG, name)
    Path(loc.path).parent.mkdir(parents=True, exist_ok=True)
    Path(loc.path).with_suffix(".safetensors").write_bytes(b"weights")
    MK.record_training(loc, name, "20210101-20230101")
    return MK.locate_for(cls, CONFIG, name)


def test_absent_then_keyed_after_training(storage):
    cls = _cls()
    assert MK.locate_for(cls, CONFIG).kind == "absent"
    loc = _train_stub(storage, cls)
    assert loc.kind == "keyed" and "/models/" in loc.path
    MK.check_manifest(loc)
    man = json.loads(MK.manifest_path(loc).read_text())
    assert man["hash"] == loc.hash and man["users"] == ["Stub"]


def test_renamed_class_with_same_inputs_finds_the_weights(storage):
    loc = _train_stub(storage, _cls("Original"), "Original")
    again = MK.locate_for(_cls("Renamed"), CONFIG, "Renamed")
    assert again.path == loc.path and again.kind == "keyed"


def test_strict_mode_raises_before_training(storage, monkeypatch):
    monkeypatch.setenv("FT_MODEL_EVAL_STRICT", "1")
    with pytest.raises(MK.ModelMissing, match="refuses to train"):
        MK.locate_for(_cls(), CONFIG)
    assert not (storage / "models").exists()


def test_legacy_directory_resolves_only_for_its_own_class(storage):
    d = storage / "OldClass"
    d.mkdir()
    (d / "OldClass.safetensors").write_bytes(b"old")
    assert MK.locate_for(_cls("OldClass"), CONFIG, "OldClass").kind == "legacy"
    assert MK.locate_for(_cls("Other"), CONFIG, "Other").kind == "absent"


def test_conflicted_hash_is_never_served(storage):
    loc = _train_stub(storage, _cls())
    idx = MK.load_index(storage)
    idx["conflicts"] = [{"hash": loc.hash, "paths": ["a", "b"], "reason": "test"}]
    MK.save_index(storage, idx)
    # the keyed directory itself still exists, so resolution falls through to it -- but the
    # index entry is not used; remove the weights to show the index alone is not served
    for f in Path(loc.path).parent.glob(f"{Path(loc.path).stem}.*"):
        if f.name != MK.MANIFEST_NAME:
            f.unlink()
    idx["entries"][loc.hash]["stem"] = "elsewhere/x"
    (storage / "elsewhere").mkdir()
    (storage / "elsewhere" / "x.safetensors").write_bytes(b"w")
    MK.save_index(storage, idx)
    assert MK.locate_for(_cls(), CONFIG).kind == "absent"


def test_manifest_mismatch_halts_and_names_fields(storage):
    loc = _train_stub(storage, _cls())
    man = json.loads(MK.manifest_path(loc).read_text())
    man["key"]["run"]["seed"] = 999
    MK.manifest_path(loc).write_text(json.dumps(man))
    with pytest.raises(MK.ModelKeyMismatch, match="run.seed"):
        MK.check_manifest(loc)


def test_second_trainer_is_refused_by_the_lock(storage):
    loc = MK.locate_for(_cls(), CONFIG)
    with MK.training_lock(loc):
        with pytest.raises(MK.ModelKeyError, match="another run is training"):
            with MK.training_lock(loc):
                pass
    with MK.training_lock(loc):  # released
        pass


def test_index_round_trip_is_atomic(storage):
    idx = {"version": 1, "entries": {"abc": {"stem": "x/y", "kind": "migrated"}}, "conflicts": []}
    MK.save_index(storage, idx)
    assert MK.load_index(storage) == idx
    assert not list(storage.glob("*.tmp"))


# ------------------------------------------------------------------------------ mainline models


def _ml(**attrs):
    return _cls("Main", MODEL_MAINLINE=True, **attrs)


def _with_wl(pairs, runmode=None):
    cfg = {"exchange": {"pair_whitelist": pairs}, "timeframe": "15m"}
    if runmode is not None:
        cfg["runmode"] = runmode
    return cfg


def test_mainline_name_ignores_pairs_window_and_seed(storage, monkeypatch):
    a = MK.locate_for(_ml(), CONFIG)
    b = MK.locate_for(_ml(), _with_wl(["BTC/USDT", "ETH/USDT", "XRP/USDT"]))
    monkeypatch.setenv("MLX_SEED", "7")
    c = MK.locate_for(_ml(), _with_wl([]))  # a dynamic pairlist configures no whitelist
    assert a.kind == b.kind == c.kind == "mainline-absent"
    assert a.path == b.path == c.path and f"/{MK.MAINLINE_SUBDIR}/" in a.path
    assert Path(a.path).parent.name.startswith(a.prefix + "__")  # readable major params first
    assert c.key["run"]["whitelist"] == MK.DYNAMIC_WHITELIST


def test_mainline_name_moves_with_every_weight_shaping_input(storage):
    base = MK.locate_for(_ml(), CONFIG).path
    assert MK.locate_for(_ml(MIN_BUY_GAIN_THRESHOLD=0.03), CONFIG).path != base   # the gains
    assert MK.locate_for(_ml(HORIZON=BaseNNStrategy.HORIZON + 1), CONFIG).path != base
    assert MK.locate_for(_ml(get_training_labels=lambda self, df: None), CONFIG).path != base
    assert MK.locate_for(_ml(), {**CONFIG, "timeframe": "1h"}).path != base
    (storage / "main_scaler.pkl").write_bytes(b"scaler-v2")
    assert MK.locate_for(_ml(), CONFIG).path != base


def test_mainline_variant_with_post_model_changes_shares_the_weights(storage):
    ign = next(n for n in MK.merged(BaseNNStrategy, "MODEL_KEY_IGNORED")
               if isinstance(MK.declared(BaseNNStrategy, n, None), (int, float)))
    v = MK.declared(BaseNNStrategy, ign)
    variant = type("MainVariant", (_ml(),), {ign: v + 1, "get_entry_conditions": lambda self, df: None})
    assert MK.locate_for(variant, CONFIG).path == MK.locate_for(_ml(), CONFIG).path


def test_a_fold_of_a_mainline_class_keeps_the_keyed_scheme(storage):
    fold = type("Main_f00", (_ml(),), {"MODEL_TRAIN_TIMERANGE": "20210101-20230101"})
    assert not MK.is_mainline(fold)
    loc = MK.locate_for(fold, CONFIG)
    assert loc.kind == "absent" and "/models/" in loc.path
    with pytest.raises(MK.ModelKeyError):
        MK.resolve(fold, _with_wl([]))  # keyed: the whitelist is still required


def test_mainline_flag_does_not_shape_the_keyed_weights(storage):
    assert MK.key_hash(MK.resolve(_ml(), CONFIG)) == _h(_cls())


def test_mainline_dry_and_live_runs_never_train(storage):
    from freqtrade.enums import RunMode

    for mode in (RunMode.DRY_RUN, RunMode.LIVE):
        with pytest.raises(MK.ModelMissing):
            MK.locate_for(_ml(), _with_wl([], mode))
    assert MK.locate_for(_ml(), _with_wl(["SOL/USDT"], RunMode.BACKTEST)).kind == "mainline-absent"


def test_mainline_training_writes_a_manifest_and_is_found_under_any_pairlist(storage):
    from freqtrade.enums import RunMode

    loc = _train_stub(storage, _ml(), "Main")
    assert loc.kind == "mainline"
    MK.check_manifest(loc)
    man = json.loads(MK.manifest_path(loc).read_text())
    assert man["kind"] == "mainline" and man["provenance"]["pairs"] == CONFIG["exchange"]["pair_whitelist"]
    live = MK.locate_for(_ml(), _with_wl([], RunMode.LIVE))
    assert live.kind == "mainline" and live.path == loc.path
    MK.check_manifest(live)


def test_mainline_manifest_of_another_shape_halts(storage):
    loc = _train_stub(storage, _ml(), "Main")
    mp = MK.manifest_path(loc)
    man = json.loads(mp.read_text())
    man["key"]["attributes"]["HORIZON"] = 999
    mp.write_text(json.dumps(man))
    with pytest.raises(MK.ModelKeyMismatch):
        MK.check_manifest(MK.locate_for(_ml(), CONFIG))


def test_adopt_mainline_copies_the_keyed_weights(storage):
    keyed = _train_stub(storage, _cls("Main"), "Main")  # trained as an ordinary keyed model
    assert keyed.kind == "keyed"
    d = MK.adopt_mainline(_ml(), CONFIG)
    loc = MK.locate_for(_ml(), _with_wl([]))
    assert loc.kind == "mainline" and Path(loc.path).parent == d
    assert Path(loc.path).with_suffix(".safetensors").read_bytes() == b"weights"
    MK.check_manifest(loc)
    man = json.loads(MK.manifest_path(loc).read_text())
    assert man["provenance"]["kind"] == "adopted" and man["provenance"]["from"] == f"{keyed.prefix}__{keyed.hash}"
    with pytest.raises(MK.ModelKeyError):
        MK.adopt_mainline(_ml(), CONFIG)  # never overwrites
