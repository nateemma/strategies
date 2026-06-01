"""
Golden-file metadata round-trip tests for the GAN backends.

These tests assert the **contract** of what each backend writes to disk
when GANInterface.save() is called — independently of whether the
underlying model actually trained or contains real weights.

Why: the existing test_gan_interface.py tests verify call-site routing
(MagicMock'd models — no file I/O) but not what ends up on disk.  When
the GAN backends are refactored to a uniform contract, these tests
catch silent metadata drift: a missing key, a renamed file, a stat that
was tracked but never persisted (the WGAN-MLX feature_min/max bug we
just diagnosed).

Each backend uses a small purpose-built stand-in class.  The stand-in
satisfies isinstance() checks inside GANInterface.save() and records
the same attributes the real class would expose, but its weight save
is a no-op so we don't need TF or MLX to run the tests.

Pickle is read back from disk directly (not via GANInterface.load,
which is itself part of what we're testing in the load suite).

Run from the strategies root:
    python -m pytest GANs/tests/test_gan_metadata_roundtrip.py -v
"""
from __future__ import annotations

import os
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Dict, Set
from unittest.mock import MagicMock

import numpy as np

# ---------------------------------------------------------------------------
# Path setup — match test_gan_interface.py's pattern.
# ---------------------------------------------------------------------------

STRATEGIES_ROOT = str(Path(__file__).parent.parent.parent)
if STRATEGIES_ROOT not in sys.path:
    sys.path.insert(0, STRATEGIES_ROOT)

# Stub freqtrade so importing strategy framework files doesn't drag it in.
_ft_stub = MagicMock()
sys.modules.setdefault("freqtrade", _ft_stub)
sys.modules.setdefault("freqtrade.persistence", _ft_stub)
sys.modules.setdefault("freqtrade.strategy", _ft_stub)

from GANs.GANType import GANType  # noqa: E402
from GANs.GANInterface import (  # noqa: E402
    GANInterface,
    GANMetadataMismatchError,
    _values_match,
    _validate_metadata_against_expected,
)


# ---------------------------------------------------------------------------
# Stand-in model classes.  Each mimics the real backend's save() interface
# enough that GANInterface.save() can route to it; isinstance checks see
# a real class (not a MagicMock); weight files are touched but not real.
# ---------------------------------------------------------------------------


class _StandInWGANMLX:
    """Stand-in for GANs.df_wgan_mlx.WGANMLX."""

    def __init__(self):
        self.num_features = 25
        self.num_classes = 3
        self.latent_dim = 64
        # Feature stats — the real WGANMLX doesn't track these today; this
        # stand-in carries them so we can assert they're persisted *if*
        # GANInterface.save chooses to write them.  The Phase-3 fix makes
        # this true.
        self.feature_mean = np.zeros(25, dtype=np.float32)
        self.feature_std = np.ones(25, dtype=np.float32)
        self.feature_min = -np.ones(25, dtype=np.float32)
        self.feature_max = np.ones(25, dtype=np.float32)

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        Path(path, "wgan_gen_mlx.safetensors").touch()


class _StandInMTWGANMLX:
    """Stand-in for GANs.df_mt_wgan_mlx.MTWGANMLX."""

    def __init__(self):
        self.seq_len = 16
        self.num_features = 25
        self.task_label_dims = {
            "trading": 3, "regime": 3, "risk": 3,
            "momentum": 3, "flow": 3, "profit": 3,
        }
        self.latent_dim = 64
        self.feature_mean = np.zeros(25, dtype=np.float32)
        self.feature_std = np.ones(25, dtype=np.float32)
        self.feature_min = -np.ones(25, dtype=np.float32)
        self.feature_max = np.ones(25, dtype=np.float32)

    def save(self, path: str, metadata: dict):
        # Real MTWGANMLX writes _s{seq_len}-suffixed files.  Replicate
        # that so the load probes (golden-file load tests) match reality.
        os.makedirs(path, exist_ok=True)
        suffix = f"_s{self.seq_len}" if self.seq_len > 1 else ""
        Path(path, f"mt_wgan_gen_mlx{suffix}.safetensors").touch()
        Path(path, f"mt_wgan_crit_mlx{suffix}.safetensors").touch()
        with open(os.path.join(path, f"mt_wgan_meta_mlx{suffix}.pkl"), "wb") as f:
            pickle.dump(metadata, f)


# ---------------------------------------------------------------------------
# Expected key contracts — single source of truth for what each backend
# MUST persist.  These are deliberately strict: a missing key fails the
# test rather than silently masking the bug.
# ---------------------------------------------------------------------------


# Constructor / model-shape keys — must always be present so the backend
# can be reconstructed from disk.
_REQUIRED_WGAN_KEYS: Set[str] = {
    "num_features",
    "num_classes",
    "latent_dim",
    "seq_len",
}

_REQUIRED_MT_WGAN_KEYS: Set[str] = {
    "seq_len",
    "num_features",
    "task_label_dims",
    "latent_dim",
}

# Stat keys — required for the postprocessing clip in generate() to
# work.  The TF path saves these; the MLX path does not (today).  We
# mark MLX cases as expectedFailure so the tests pass on the current
# tree but flip to "unexpected pass" once Phase 3 fixes the gap.
_FEATURE_STAT_KEYS: Set[str] = {
    "feature_mean",
    "feature_std",
    "feature_min",
    "feature_max",
}

# Any extra_metadata kwargs the caller passes (typically MASTER thresholds
# from CreateGANBase) must always round-trip.
_USER_EXTRAS: Dict[str, object] = {
    "min_buy_gain_threshold": 0.016,
    "min_sell_loss_threshold": 0.012,
    "training_type": 19,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _save_via_interface(
    gan_type: GANType,
    model: object,
    save_path: str,
    *,
    backend_cls,
    **extras,
) -> None:
    """
    Drive GANInterface.save() with a stand-in model wrapped in a real
    GANBackend instance.

    Phase 2 of the GAN refactor moved per-type save logic from
    isinstance-based dispatch in GANInterface into concrete backend
    classes.  The cleanest way to test "what does save write to disk
    for backend X" is to construct backend X with a stand-in ``_model``
    and have GANInterface delegate to it.
    """
    backend = backend_cls()
    backend._model = model

    iface = GANInterface(gan_type, save_path=save_path, prefer_mlx=True)
    iface._backend = backend
    iface.save(**extras)


def _load_pickle(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# WGAN (single-task) — MLX path
# ---------------------------------------------------------------------------


class TestWGANMLXMetadata(unittest.TestCase):
    """GANInterface.save() for WGAN MLX writes wgangp_metadata.pkl."""

    def _save_and_load_metadata(self, save_path: str) -> dict:
        from GANs.backends.wgan import WGANMLXBackend  # noqa: E402
        model = _StandInWGANMLX()
        _save_via_interface(
            GANType.WGAN, model, save_path,
            backend_cls=WGANMLXBackend,
            **_USER_EXTRAS,
        )
        meta_path = os.path.join(save_path, "wgangp_metadata.pkl")
        self.assertTrue(
            os.path.exists(meta_path),
            f"Expected wgangp_metadata.pkl to exist at {meta_path}",
        )
        return _load_pickle(meta_path)

    def test_required_keys_present(self):
        with tempfile.TemporaryDirectory() as td:
            meta = self._save_and_load_metadata(td)
        missing = _REQUIRED_WGAN_KEYS - set(meta.keys())
        self.assertFalse(
            missing,
            f"WGAN MLX metadata missing required keys: {missing}",
        )

    def test_required_values_match_model(self):
        with tempfile.TemporaryDirectory() as td:
            meta = self._save_and_load_metadata(td)
        self.assertEqual(meta["num_features"], 25)
        self.assertEqual(meta["num_classes"], 3)
        self.assertEqual(meta["latent_dim"], 64)
        self.assertEqual(meta["seq_len"], 1)

    def test_user_extras_round_trip(self):
        with tempfile.TemporaryDirectory() as td:
            meta = self._save_and_load_metadata(td)
        for key, expected in _USER_EXTRAS.items():
            self.assertIn(key, meta, f"missing user extra {key!r}")
            self.assertEqual(meta[key], expected, f"user extra {key!r} mutated")

    def test_feature_stats_persisted(self):
        with tempfile.TemporaryDirectory() as td:
            meta = self._save_and_load_metadata(td)
        missing = _FEATURE_STAT_KEYS - set(meta.keys())
        self.assertFalse(
            missing,
            f"WGAN MLX metadata missing feature stats: {missing}",
        )


# ---------------------------------------------------------------------------
# MT_WGAN (multi-task) — MLX path
# ---------------------------------------------------------------------------


class TestMTWGANMLXMetadata(unittest.TestCase):
    """GANInterface.save() for MT_WGAN MLX writes mt_wgan_meta_mlx_s{N}.pkl."""

    def _save_and_load_metadata(self, save_path: str) -> dict:
        from GANs.backends.mt_wgan import MTWGANMLXBackend  # noqa: E402
        model = _StandInMTWGANMLX()
        _save_via_interface(
            GANType.MT_WGAN, model, save_path,
            backend_cls=MTWGANMLXBackend,
            **_USER_EXTRAS,
        )
        # seq_len=16 ⇒ filename gets _s16 suffix
        meta_path = os.path.join(save_path, "mt_wgan_meta_mlx_s16.pkl")
        self.assertTrue(
            os.path.exists(meta_path),
            f"Expected mt_wgan_meta_mlx_s16.pkl to exist at {meta_path}",
        )
        return _load_pickle(meta_path)

    def test_required_keys_present(self):
        with tempfile.TemporaryDirectory() as td:
            meta = self._save_and_load_metadata(td)
        missing = _REQUIRED_MT_WGAN_KEYS - set(meta.keys())
        self.assertFalse(
            missing,
            f"MT_WGAN MLX metadata missing required keys: {missing}",
        )

    def test_required_values_match_model(self):
        with tempfile.TemporaryDirectory() as td:
            meta = self._save_and_load_metadata(td)
        self.assertEqual(meta["seq_len"], 16)
        self.assertEqual(meta["num_features"], 25)
        self.assertEqual(meta["latent_dim"], 64)
        self.assertEqual(
            meta["task_label_dims"],
            {"trading": 3, "regime": 3, "risk": 3,
             "momentum": 3, "flow": 3, "profit": 3},
        )

    def test_user_extras_round_trip(self):
        with tempfile.TemporaryDirectory() as td:
            meta = self._save_and_load_metadata(td)
        for key, expected in _USER_EXTRAS.items():
            self.assertIn(key, meta, f"missing user extra {key!r}")
            self.assertEqual(meta[key], expected, f"user extra {key!r} mutated")

    def test_feature_stats_persisted(self):
        with tempfile.TemporaryDirectory() as td:
            meta = self._save_and_load_metadata(td)
        missing = _FEATURE_STAT_KEYS - set(meta.keys())
        self.assertFalse(
            missing,
            f"MT_WGAN MLX metadata missing feature stats: {missing}",
        )


# ---------------------------------------------------------------------------
# CTAB-GAN — uses model.save() directly via GANInterface (class-style API)
# ---------------------------------------------------------------------------


class TestCTABGANMetadataDelegation(unittest.TestCase):
    """
    For CTAB-GAN family GANInterface.save() simply calls
    ``self._model.save(self.save_path, **extra_metadata)``.  We can
    verify the call shape with a MagicMock, since the actual file write
    is the model's responsibility (covered in the model-level tests in
    test_quality_suite / test_functional_suite).
    """

    def _drive(self, gan_type: GANType, **extras):
        mock_model = MagicMock()
        iface = GANInterface(gan_type, save_path="/tmp/ctab", prefer_mlx=False)
        iface._model = mock_model
        iface.save(**extras)
        return mock_model

    def test_ctab_gan_save_passes_save_path_and_extras(self):
        model = self._drive(GANType.CTAB_GAN, **_USER_EXTRAS)
        model.save.assert_called_once_with("/tmp/ctab", **_USER_EXTRAS)

    def test_mt_ctab_gan_save_passes_save_path_and_extras(self):
        model = self._drive(GANType.MT_CTAB_GAN, **_USER_EXTRAS)
        model.save.assert_called_once_with("/tmp/ctab", **_USER_EXTRAS)


# ---------------------------------------------------------------------------
# Cross-backend invariants — properties that should hold for every type
# ---------------------------------------------------------------------------


class TestCrossBackendInvariants(unittest.TestCase):
    """
    Properties any GAN-backend metadata must obey, regardless of type.
    These exist so that when a new backend is added, the test author
    has a checklist of what the metadata must contain.
    """

    def test_user_extras_are_pickle_safe(self):
        """If a caller passes np.float32 / np.int64 in extras, they
        round-trip through pickle as plain Python types or, at minimum,
        survive a load → equality check.  Belt-and-suspenders for the
        TrackedDict-style sanitisation gotchas the MT_WGAN TF path has."""
        extras = {
            "min_buy_gain_threshold": np.float32(0.016),
            "training_type": np.int64(19),
        }
        with tempfile.TemporaryDirectory() as td:
            from GANs.backends.wgan import WGANMLXBackend  # noqa: E402
            model = _StandInWGANMLX()
            _save_via_interface(
                GANType.WGAN, model, td,
                backend_cls=WGANMLXBackend,
                **extras,
            )
            meta = _load_pickle(os.path.join(td, "wgangp_metadata.pkl"))

        self.assertAlmostEqual(float(meta["min_buy_gain_threshold"]), 0.016, places=5)
        self.assertEqual(int(meta["training_type"]), 19)


# ---------------------------------------------------------------------------
# Metadata validation — GANInterface.load(expected=...) must raise on drift
# ---------------------------------------------------------------------------


class TestMetadataValidationHelpers(unittest.TestCase):
    """Pure-function tests for the value-comparison helper."""

    def test_floats_compared_with_tolerance(self):
        """np.float32 round-trip drift must not register as mismatch."""
        self.assertTrue(_values_match(np.float32(0.016), 0.016))
        self.assertTrue(_values_match(0.016, np.float32(0.016)))

    def test_floats_strict_difference_fails(self):
        self.assertFalse(_values_match(0.016, 0.020))

    def test_int_eq(self):
        self.assertTrue(_values_match(np.int64(19), 19))
        self.assertFalse(_values_match(19, 20))

    def test_lists_order_sensitive(self):
        self.assertTrue(_values_match(["a", "b", "c"], ["a", "b", "c"]))
        self.assertFalse(_values_match(["a", "c", "b"], ["a", "b", "c"]))

    def test_lists_length_mismatch(self):
        self.assertFalse(_values_match(["a", "b"], ["a", "b", "c"]))

    def test_dicts_compared_recursively(self):
        self.assertTrue(_values_match(
            {"trading": 3, "regime": 3},
            {"trading": 3, "regime": 3},
        ))
        self.assertFalse(_values_match(
            {"trading": 3, "regime": 3},
            {"trading": 3, "regime": 5},
        ))


class TestMetadataValidatorRaises(unittest.TestCase):
    """``_validate_metadata_against_expected`` raises on any drift."""

    def test_passes_when_all_match(self):
        # No exception → success.
        _validate_metadata_against_expected(
            saved={"a": 1, "b": 2.0},
            expected={"a": 1, "b": 2.0},
            gan_type=GANType.WGAN,
            save_path="/tmp",
        )

    def test_raises_on_value_mismatch(self):
        with self.assertRaises(GANMetadataMismatchError) as ctx:
            _validate_metadata_against_expected(
                saved={"a": 1, "b": 2.0},
                expected={"a": 1, "b": 9.9},
                gan_type=GANType.WGAN,
                save_path="/tmp",
            )
        self.assertIn("b", ctx.exception.mismatches)
        self.assertEqual(ctx.exception.mismatches["b"], (2.0, 9.9))

    def test_missing_keys_warn_and_do_not_raise(self):
        """Backward-compatibility: a saved-metadata key that's absent is
        emitted as a warning and validation proceeds. This is the
        deliberate 2026-05-30 behavior change so older GAN saves that
        predate a newly-added validator key still load. See
        feedback_*_missing → warn-not-raise note."""
        with self.assertLogs(level="WARNING") as logs:
            # No raise expected.
            _validate_metadata_against_expected(
                saved={"a": 1},
                expected={"a": 1, "min_buy_gain_threshold": 0.016},
                gan_type=GANType.WGAN,
                save_path="/tmp",
            )
        # The warning should mention the missing key and its expected value.
        warning_text = "\n".join(logs.output)
        self.assertIn("min_buy_gain_threshold", warning_text)
        self.assertIn("0.016", warning_text)

    def test_value_mismatch_still_raises_even_when_other_key_missing(self):
        """The lenient missing-key treatment must NOT downgrade real
        value mismatches. If one key is missing (warn) and another has a
        wrong value (raise), the raise still wins — only mismatched
        values surface in the exception."""
        with self.assertRaises(GANMetadataMismatchError) as ctx:
            with self.assertLogs(level="WARNING"):
                _validate_metadata_against_expected(
                    saved={"a": 1, "training_type": 1},
                    expected={
                        "a": 1,
                        "min_buy_gain_threshold": 0.016,  # missing → warn
                        "training_type": 17,              # mismatched → raise
                    },
                    gan_type=GANType.WGAN,
                    save_path="/tmp",
                )
        self.assertIn("training_type", ctx.exception.mismatches)
        self.assertNotIn(
            "min_buy_gain_threshold", ctx.exception.mismatches,
            "missing keys must not be reported as mismatches",
        )

    def test_extra_saved_keys_are_allowed(self):
        """Saved metadata may include extra keys beyond what the caller
        cares about — validation only checks what was passed in."""
        _validate_metadata_against_expected(
            saved={"a": 1, "b": 2, "c": 3},
            expected={"a": 1},
            gan_type=GANType.WGAN,
            save_path="/tmp",
        )

    def test_error_message_lists_every_mismatch(self):
        with self.assertRaises(GANMetadataMismatchError) as ctx:
            _validate_metadata_against_expected(
                saved={"a": 1, "b": 2, "c": 3},
                expected={"a": 1, "b": 99, "c": 100},
                gan_type=GANType.WGAN,
                save_path="/tmp/some/path",
            )
        msg = str(ctx.exception)
        self.assertIn("b", msg)
        self.assertIn("c", msg)
        self.assertIn("/tmp/some/path", msg)
        self.assertIn("WGAN", msg)


class TestGANInterfaceLoadValidation(unittest.TestCase):
    """GANInterface.load(expected=...) — validation path.

    These tests stub out ``load_with_fallback`` (the registry probe that
    actually opens files and instantiates a backend) so we can drive
    GANInterface.load with a known-good metadata payload and assert the
    expected-vs-saved comparison runs.  This avoids the TF/MLX dependency
    that the on-disk round-trip tests carry, while still exercising the
    full code path through ``GANInterface.load``."""

    def _patch_load_with_fallback(self, fake_metadata):
        """Return a context manager that patches load_with_fallback to
        return ``(stub_backend, fake_metadata)`` regardless of inputs."""
        from unittest.mock import patch

        stub_backend = MagicMock()
        stub_backend._model = MagicMock()

        return patch(
            "GANs.GANInterface.load_with_fallback",
            return_value=(stub_backend, fake_metadata),
        )

    def test_load_returns_metadata_when_expected_matches(self):
        """Saved == expected → load() returns the metadata dict, no raise."""
        saved_metadata = {
            "num_features": 25,
            "num_classes": 3,
            "min_buy_gain_threshold": 0.016,
            "training_type": 19,
        }
        with self._patch_load_with_fallback(saved_metadata):
            iface = GANInterface(GANType.WGAN, save_path="/tmp", prefer_mlx=True)
            result = iface.load(expected={
                "min_buy_gain_threshold": 0.016,
                "training_type": 19,
            })
        # Returned dict carries the full saved metadata, not just the
        # subset we validated against.
        self.assertEqual(result["num_features"], 25)
        self.assertEqual(int(result["training_type"]), 19)

    def test_load_raises_on_threshold_mismatch(self):
        """Saved threshold drifts from expected → GANMetadataMismatchError.

        This is the contract that prevents the silent-override bug:
        previously CTAB-GAN's load() would overwrite
        ``self.MIN_BUY_GAIN_THRESHOLD`` with the GAN's stale value.
        Now the strategy must declare its expected thresholds and a
        mismatch is surfaced loudly."""
        saved_metadata = {
            "min_buy_gain_threshold": 0.016,
            "training_type": 19,
        }
        with self._patch_load_with_fallback(saved_metadata):
            iface = GANInterface(GANType.WGAN, save_path="/tmp", prefer_mlx=True)
            with self.assertRaises(GANMetadataMismatchError) as ctx:
                iface.load(expected={
                    "min_buy_gain_threshold": 0.030,  # drifted
                    "training_type": 19,
                })
        self.assertIn("min_buy_gain_threshold", ctx.exception.mismatches)
        self.assertNotIn("training_type", ctx.exception.mismatches)

    def test_load_warns_when_expected_key_absent_from_saved(self):
        """Backward-compatibility: a strategy that adds a new metadata key
        DOES load models that predate it; the absence is emitted as a
        warning and load proceeds. This is the deliberate 2026-05-30
        behavior change."""
        saved_metadata = {"num_features": 25}  # no thresholds saved
        with self._patch_load_with_fallback(saved_metadata):
            iface = GANInterface(GANType.WGAN, save_path="/tmp", prefer_mlx=True)
            with self.assertLogs(level="WARNING") as logs:
                # No raise expected.
                metadata = iface.load(expected={"min_buy_gain_threshold": 0.016})
        warning_text = "\n".join(logs.output)
        self.assertIn("min_buy_gain_threshold", warning_text)
        self.assertEqual(metadata["num_features"], 25)

    def test_load_no_expected_skips_validation(self):
        """Backwards-compat: load() without expected= just returns metadata."""
        saved_metadata = {"num_features": 25, "min_buy_gain_threshold": 0.016}
        with self._patch_load_with_fallback(saved_metadata):
            iface = GANInterface(GANType.WGAN, save_path="/tmp", prefer_mlx=True)
            # No expected= → no raise, returns the saved dict.
            result = iface.load()
        self.assertEqual(result["num_features"], 25)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
