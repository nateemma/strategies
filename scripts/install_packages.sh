#!/bin/zsh

# script to install additional packages for the strategies in this repo

# I find this useful in that I normally have to do this if I upgrade something fundamental like python or anaconda.
# Also, I sometimes get in a mess with package dependencies and have to reset
# Either of these requires me to do a full install of freqtrade (sh setup.sh -r), and then re-install these packages

# Notes:
# - if you need conda versions of packages, you'll need to install those from within the freqtrade venv
# - at the current time, conda packages for the M1 Mac are a mess of conflicting version requirements, so beware
# - the neural network-based strategies require either keras & tensorflow, or darts & pytorch
#
# IMPORTANT — Apple Silicon GPU support requires a specific version triple:
#   * Python 3.11 or 3.12       (tensorflow-metal has no wheels for 3.13+)
#   * tensorflow == 2.17.0      (2.18+ is currently broken with tensorflow-metal)
#   * tensorflow-metal == 1.2.0 (latest as of writing)
# Anything outside this combination will silently fall back to CPU on M-series Macs.

# --- Pinned versions --------------------------------------------------------
TF_VERSION="2.17.0"
TF_METAL_VERSION="1.2.0"
# ---------------------------------------------------------------------------

# Sanity-check the active Python version before doing anything
PY_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null)
case "$PY_VERSION" in
  3.11|3.12)
    echo "Python $PY_VERSION detected — compatible with tensorflow-metal $TF_METAL_VERSION"
    ;;
  *)
    echo "WARNING: Python $PY_VERSION detected. tensorflow-metal $TF_METAL_VERSION only ships wheels for Python 3.11 / 3.12."
    echo "         The Apple Silicon GPU path will fail. Recreate the venv with python@3.11 or python@3.12 first."
    ;;
esac
echo ""

# function to get y/n answer. Pass the prompt as arg
prompt_user () {
  result=0

  read -rq "yn?${1} (y/n) " # zsh-specific

  if [ "$yn" = 'y' ]; then
    result=1
  else
    result=0
  fi
  echo $result # stupid zsh doesn't really have a return
}

# Handle environment detection
# If we are in the freqtrade repo, try to use the local .venv
SCRIPT_DIR=$(dirname "$0")
REPO_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)

if [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
    echo "Detected Freqtrade virtual environment at $REPO_ROOT/.venv"
    # Use the venv's pip directly to be sure
    alias pip="$REPO_ROOT/.venv/bin/pip"
    alias pip3="$REPO_ROOT/.venv/bin/pip"
    echo "Using venv pip: $($REPO_ROOT/.venv/bin/pip --version)"
fi

# update installation tools first
echo "Updating pip, setuptools, and wheel..."
pip install --upgrade pip setuptools wheel

# Python 3.12+ removed `distutils` from the stdlib (PEP 632), but TensorFlow 2.17 still
# does `import distutils as _distutils` at the top of its __init__.py. Setuptools ships
# a meta-path finder shim, but it doesn't always fire early enough under freqtrade's
# strategy resolver. The bulletproof fix is to drop a real `distutils/` package into
# site-packages that delegates to setuptools._distutils — no shim machinery required.
if [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
  SITE_PKGS=$($REPO_ROOT/.venv/bin/python -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null)
  if [ -n "$SITE_PKGS" ] && [ ! -f "$SITE_PKGS/distutils/__init__.py" ]; then
    echo "Installing distutils compatibility shim (Python 3.12+ + TensorFlow 2.17)..."
    mkdir -p "$SITE_PKGS/distutils"
    cat > "$SITE_PKGS/distutils/__init__.py" << 'PYEOF'
"""
distutils compatibility shim for Python 3.12+.

Python 3.12 removed distutils from the stdlib (PEP 632). This shim makes
`import distutils` (and `import distutils.sysconfig` etc.) keep working by
delegating to setuptools's vendored copy in `setuptools._distutils`.

Required because TensorFlow 2.17 still does `import distutils as _distutils`
at the top of its __init__.py.
"""
import sys as _sys
from setuptools import _distutils as _real_distutils
_sys.modules[__name__] = _real_distutils
PYEOF
  fi

  # Belt-and-braces: also ensure setuptools is imported at startup so its
  # _distutils_hack is registered. Cheap insurance even with the shim above.
  if [ -n "$SITE_PKGS" ] && [ ! -f "$SITE_PKGS/sitecustomize.py" ]; then
    echo "Installing sitecustomize.py..."
    cat > "$SITE_PKGS/sitecustomize.py" << 'PYEOF'
"""Force setuptools import at startup so its distutils hack is registered.
Belt-and-braces fix for Python 3.12+ + TensorFlow 2.17 (see distutils/ shim)."""
import setuptools  # noqa: F401
PYEOF
  fi
fi

# install generally used packages
# Note: tensorflow / keras are intentionally NOT in this list — they are pinned
# alongside tensorflow-metal in the next section so versions stay coordinated.
# Note: pandas-ta is installed separately below — the upstream 0.4.x line on PyPI
# requires numpy>=2.2.6 (incompatible with our TF 2.17 + numpy<2 pin), AND its older
# 0.3.x releases were removed from PyPI. We install MerlinR's preserved fork instead,
# which keeps the original `import pandas_ta` name.
pkg_general=("finta" "prettytable" "PyWavelets" "simdkalman" "pykalman" "scipy" "scikit-learn" \
"ast_comments" "rich" "xgboost" "lightgbm" "statsmodels" "imblearn" \
"pytest" "pytest-mock" "pytest-xdist" "pytest-asyncio")

if [[ $(prompt_user "Install general packages?: ") -eq 1 ]]; then
  echo ""
  for pkg in $pkg_general; do
    pip3 install $pkg
  done

  # Install pandas-ta from MerlinR's preserved fork (keeps the `pandas_ta` import name)
  echo "Installing pandas-ta (MerlinR fork — preserves 0.3.14b API)..."
  pip3 install "pandas_ta @ git+https://github.com/MerlinR/Pandas-ta-fork.git"

  # Force numpy < 2.0 to avoid the "umath failed to import" error
  # This is critical for compatibility with TensorFlow 2.17 and other legacy compiled modules.
  echo "Enforcing NumPy < 2.0..."
  pip3 install "numpy<2"
fi
echo ""

# install packages for tensorflow-based strategies (MacOS-specific)

if [[ $(prompt_user "Install tensorflow packages?: ") -eq 1 ]]; then

  # check whether this uses an Apple CPU
  cpu_brand=$(sysctl -n machdep.cpu.brand_string)
  if [[ $cpu_brand == Apple* ]]; then
      echo ""
      echo "Detected Apple Silicon CPU: $cpu_brand"
      echo "Installing TensorFlow ${TF_VERSION} + tensorflow-metal ${TF_METAL_VERSION} for Apple Silicon..."
      # tensorflow-macos was deprecated as of TF 2.13+; the unified `tensorflow` wheel runs
      # natively on arm64 macOS. We pin to 2.17.0 because tensorflow-metal 1.2.0 is broken
      # against TF 2.18+ (https://github.com/tensorflow/tensorflow/issues/84167).
      pip3 install "tensorflow==${TF_VERSION}"
      pip3 install "tensorflow-metal==${TF_METAL_VERSION}"
      # Don't `--upgrade keras` here: TF 2.17 pulls the matching Keras 3.x release.
      # Upgrading separately can silently move keras past the version TF was built against.
      pip3 install --upgrade pandas
      pip3 install mlx
  else
    echo "Installing standard TensorFlow ${TF_VERSION}..."
    pip3 install "tensorflow==${TF_VERSION}"
    pip3 install --upgrade pandas
  fi

  # Re-enforce NumPy < 2 in case TF's resolver pulled a 2.x build
  echo "Re-enforcing NumPy < 2.0 after TensorFlow install..."
  pip3 install "numpy<2"

  # Verify the GPU is actually visible (Apple Silicon only)
  if [[ $cpu_brand == Apple* ]]; then
    echo ""
    echo "Verifying Metal GPU detection..."
    python3 -c "import tensorflow as tf; gpus = tf.config.list_physical_devices('GPU'); print(f'TF {tf.__version__} — GPUs: {gpus}'); exit(0 if gpus else 1)" \
      && echo "GPU detected — tensorflow-metal is wired up correctly." \
      || echo "WARNING: TensorFlow loaded but no GPU was detected. Check Python version (must be 3.11 or 3.12) and tensorflow-metal install."
  fi
fi
echo ""

# install packages for darts/pytorch-based strategies

if [[ $(prompt_user "Install darts/pytorch packages?: ") -eq 1 ]]; then
  echo ""
  conda install pytorch torchvision -c pytorch
  # pip3 install darts
  pip3 install "u8darts[all]"
  pip3 install statsforecast
  pip3 install multiprocess
fi
echo ""