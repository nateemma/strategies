"""Contract test for the GAN scorecard driver.

NOT a quality assertion -- it asserts the DRIVER behaves, specifically that it
produces a row for EVERY registered (type, backend) pair including the ones that
fail. A table that silently omits its broken entries is exactly the table that
hides the problem, and this repo has a documented history of changes that run
cleanly while doing nothing.

Quality bars live in test_quality_suite.py; ranked numbers live in
docs/GAN_SCORECARD.md.

The full sweep fits real models, so it is gated:
    RUN_SLOW_TESTS=1 python -m pytest user_data/strategies/GANs/tests/test_scorecard.py -v
Un-gated tests below cover the fixture and renderer only.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from GANs.GANType import GANType
from GANs.quality.scorecard import (ADAPTERS, build_scorecard, make_fixture,
                                    render_markdown, score_one)

RUN_SLOW = os.environ.get("RUN_SLOW_TESTS") == "1"


class TestFixtureAndRenderer(unittest.TestCase):
    """Fast checks -- no model fitting."""

    def test_fixture_is_class_discriminative(self):
        """If the fixture had no class structure the utility probe could not
        distinguish a good GAN from a bad one."""
        x, y = make_fixture(n=400)
        self.assertEqual(len(x), 400)
        self.assertEqual(y.shape[1], 3)
        cls = y.argmax(1)
        self.assertGreater(len(np.unique(cls)), 1)
        m0, m1 = x[cls == 0].mean(0), x[cls == 1].mean(0)
        self.assertGreater(float(np.abs(m0 - m1).max()), 0.3)

    def test_every_non_trivial_type_has_an_adapter(self):
        """A missing adapter silently yields an empty row -- catch it here."""
        for t in GANType:
            if t.name in ("NONE", "BOTH"):
                continue
            self.assertIn(t.name, ADAPTERS, f"no scorecard adapter for {t.name}")

    def test_renderer_emits_a_row_per_input(self):
        rows = [{"type": "X", "backend": "TF", "status": "ok"},
                {"type": "Y", "backend": "MLX", "status": "fit-failed"}]
        md = render_markdown(rows)
        self.assertEqual(len(md.strip().split("\n")), 4)   # header + sep + 2
        self.assertIn("fit-failed", md)

    def test_renderer_tolerates_missing_keys(self):
        self.assertIn("|", render_markdown([{"type": "X"}]))


@unittest.skipUnless(RUN_SLOW, "set RUN_SLOW_TESTS=1 -- fits real models")
class TestDriverContract(unittest.TestCase):
    def test_every_registered_pair_yields_a_row(self):
        x, y = make_fixture(n=300, seed=1)
        rows = build_scorecard(x, y)
        seen = {(r["type"], r["backend"]) for r in rows}
        for t in GANType:
            if t.name in ("NONE", "BOTH"):
                continue
            for be in ("TF", "MLX"):
                self.assertIn((t.name, be), seen, f"{t.name}/{be} produced no row")
        for r in rows:
            self.assertTrue(r.get("status"), f"{r['type']}/{r['backend']} has no status")

    def test_a_broken_backend_becomes_a_row_not_an_exception(self):
        """The driver must never abort the sweep."""
        x, y = make_fixture(n=120, seed=2)
        bad = np.full_like(x, np.nan)
        row = score_one(GANType.WGAN, True, bad, y)
        self.assertTrue(row["status"])
        self.assertNotEqual(row["status"], "ok")


if __name__ == "__main__":
    unittest.main()
