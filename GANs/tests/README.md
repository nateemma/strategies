# GAN Test Suite

All commands below are run from the **freqtrade root directory** (`~/freqtrade/`).

## Test files

| File | What it covers |
|---|---|
| `test_gan_interface.py` | `GANInterface` contract — routing, defaults, overrides, error handling. No TF/GPU required (fully mocked). |
| `test_functional_suite.py` | Output contract, save/load, and end-to-end integration for every GAN type. Requires TF. |
| `test_quality_suite.py` | Trains each GAN and checks that generated data meets statistical quality bars. Slow (5–15 min). Requires `RUN_SLOW_TESTS=1`. |

---

## Running the tests

### Interface tests (fast, no GPU)

```bash
python -m pytest user_data/strategies/GANs/tests/test_gan_interface.py -v
```

### Functional tests (all GAN types)

```bash
python -m pytest user_data/strategies/GANs/tests/test_functional_suite.py -v
```

Run a single GAN type (e.g. WGAN):

```bash
python -m pytest user_data/strategies/GANs/tests/test_functional_suite.py -k "WGAN" -v
```

Available type names: `WGAN`, `MTWGAN`, `CGAN`, `CTABGAN`, `MTCTABGAN`, `TabDDPM`

Each type generates three test classes (e.g. `TestWGANFitGenContract`, `TestWGANFitGenSaveLoad`, `TestWGANFitGenInterface`). Run one class directly:

```bash
python -m pytest "user_data/strategies/GANs/tests/test_functional_suite.py::TestWGANFitGenSaveLoad" -v
```

### Quality tests (slow — trains real models)

Quality tests are gated behind `RUN_SLOW_TESTS=1` and take 5–15 minutes on CPU.

Run all quality tests:

```bash
RUN_SLOW_TESTS=1 python -m pytest user_data/strategies/GANs/tests/test_quality_suite.py -v
```

Run a single GAN type:

```bash
RUN_SLOW_TESTS=1 python -m pytest "user_data/strategies/GANs/tests/test_quality_suite.py::TestWGANQuality" -v
```

Available type names: `TestWGANQuality`, `TestMTWGANQuality`, `TestCTABGANQuality`, `TestMTCTABGANQuality`, `TestTabDDPMQuality`

### All tests at once

```bash
RUN_SLOW_TESTS=1 python -m pytest user_data/strategies/GANs/tests/ -v
```

---

## Using unittest instead of pytest

```bash
# Interface
python -m unittest GANs.tests.test_gan_interface -v

# Functional
python -m unittest GANs.tests.test_functional_suite -v

# Quality
RUN_SLOW_TESTS=1 python -m unittest GANs.tests.test_quality_suite -v

# Single class
RUN_SLOW_TESTS=1 python -m unittest GANs.tests.test_quality_suite.TestWGANQuality -v
```

Run from `user_data/strategies/` when using the `GANs.tests.*` module paths.

---

## Notes

- **xdist is disabled** for this directory (`conftest.py`). GAN tests clear TF session state in `setUpClass`; running them across xdist workers would break that isolation.
- **CTAB-GAN quality tests** skip `test_label_fidelity_above_chance`. CTAB-GAN is a general tabular synthesiser; its quality is measured by statistical fidelity (RMSE, range coverage), not discriminative class conditioning.
- **GPU/MPS**: Tests suppress GPU/MPS via environment variables (`TF_DISABLE_MPS=1`, `CUDA_VISIBLE_DEVICES=""`). To enable GPU, unset those before running.
