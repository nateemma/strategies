#!/usr/bin/env bash
#
# Upgrade freqtrade without destroying this repo's ML environment.
#
# Why this exists (do NOT use freqtrade's own setup.sh for this):
#   * `setup.sh -i` calls reset() -> recreate_environments() -> `rm -rf .venv`,
#     wiping every strategy package. It also does NOT git pull, so it silently
#     fails to upgrade anything.
#   * `setup.sh` hardcodes PYTHON="python3.13" whenever `uv` is installed,
#     ignoring pyenv. tensorflow-metal ships no cp313 wheel, so that rebuild
#     silently kills Apple Silicon GPU support.
# Patching setup.sh is not an option: it is an upstream-tracked file and any
# local edit conflicts on the next `git pull`. Hence this script, kept here.
#
# Two pinned constraints this script exists to defend:
#   PY_VERSION=3.12  - tensorflow-metal 1.2.0 has wheels for cp39/310/311/312 only.
#   numpy<2          - tensorflow 2.17 requires numpy<2.0.0 (ABI). freqtrade's
#                      metadata asks for numpy>2.0, but that is a resolver
#                      warning only: freqtrade runs correctly on numpy 1.26.x.
#                      A plain upgrade re-pulls numpy 2.x and breaks TF, so we
#                      always re-pin afterwards.
#
# Usage:
#   ./upgrade_freqtrade.sh              # in-place upgrade (git pull + deps + re-pin)
#   ./upgrade_freqtrade.sh --rebuild    # destroy and recreate .venv from scratch
#   ./upgrade_freqtrade.sh --no-pull    # skip git pull, just repair the venv
set -euo pipefail

PY_VERSION="${FT_PYTHON_VERSION:-3.12}"
# Constraints freqtrade's requirements-dev.txt violates and we must restore after
# every dependency install. Each is a resolver warning only - freqtrade runs fine.
#   numpy<2      : tensorflow 2.17 ABI (freqtrade asks for numpy>2.0)
#   protobuf<5   : tensorflow 2.17 caps protobuf, but freqtrade pins
#                  tensorboard==2.21.0 which drags in protobuf>=6.31.1. Symptom:
#                  "MessageFactory object has no attribute GetPrototype".
#   tensorboard  : must match TF (>=2.17,<2.18); freqtrade only needs it for
#                  freqai-rl, which this repo does not use.
TF_PINS=("numpy<2" "protobuf<5" "tensorboard>=2.17,<2.18")

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
VENV="$REPO_ROOT/.venv"
REBUILD=false; DO_PULL=true
for a in "$@"; do
  case "$a" in
    --rebuild) REBUILD=true ;;
    --no-pull) DO_PULL=false ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "unknown option: $a" >&2; exit 2 ;;
  esac
done

say() { printf '\n\033[1m==> %s\033[0m\n' "$1"; }
cd "$REPO_ROOT"

# ---------------------------------------------------------------- git pull ---
if [ "$DO_PULL" = true ]; then
  say "Updating freqtrade source"
  if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    echo "ERROR: outer repo has modified tracked files. This repo must stay pristine." >&2
    git status --porcelain --untracked-files=no >&2
    exit 1
  fi
  before=$(git rev-parse --short HEAD)
  git pull --ff-only
  echo "  $before -> $(git rev-parse --short HEAD)"
fi

# -------------------------------------------------------------------- venv ---
PYBIN="python${PY_VERSION}"
command -v "$PYBIN" >/dev/null 2>&1 || { echo "ERROR: $PYBIN not found (pyenv install ${PY_VERSION}.x)" >&2; exit 1; }

if [ "$REBUILD" = true ] || [ ! -x "$VENV/bin/python" ]; then
  say "Creating venv on Python $PY_VERSION"
  rm -rf "$VENV"
  if command -v uv >/dev/null 2>&1; then uv venv "$VENV" --python "$PY_VERSION"
  else "$PYBIN" -m venv "$VENV"; fi
else
  cur=$("$VENV/bin/python" -c 'import sys;print(f"{sys.version_info.major}.{sys.version_info.minor}")')
  if [ "$cur" != "$PY_VERSION" ]; then
    echo "ERROR: .venv is Python $cur, expected $PY_VERSION. Re-run with --rebuild." >&2
    exit 1
  fi
  say "Reusing existing venv (Python $cur)"
fi

PIP="$VENV/bin/pip"
# uv pip takes --python on the subcommand, not the group; VIRTUAL_ENV is simpler
# and is what uv reads to target an existing venv.
if command -v uv >/dev/null 2>&1; then PIP="uv pip"; export VIRTUAL_ENV="$VENV"; fi

# ---------------------------------------------------------------- freqtrade ---
say "Installing freqtrade + dev requirements"
$PIP install --upgrade pip setuptools wheel
$PIP install --upgrade -r requirements-dev.txt
$PIP install -e .

if [ "$REBUILD" = true ]; then
  say "Installing strategy packages"
  echo "  run scripts/install_packages.sh yourself for the full ML stack;"
  echo "  installing the non-interactive subset here."
  # NOTE: pandas_ta is intentionally absent - freqtrade now ships ft-pandas-ta,
  # which provides the `pandas_ta` import name. Installing the MerlinR fork on
  # top would shadow the maintained one.
  # NOTE: darts/statsforecast are intentionally absent - they pin pandas<3.0.0,
  # which conflicts with freqtrade's pandas 3.x. Installing them downgrades
  # pandas and breaks freqtrade.
  $PIP install finta prettytable PyWavelets simdkalman pykalman imblearn \
               ast_comments rich xgboost lightgbm statsmodels
  $PIP install "tensorflow==2.17.0" "tensorflow-metal==1.2.0" mlx
fi

# ------------------------------------------- restore TF pins (see TF_PINS) ---
say "Restoring tensorflow-compatible pins: ${TF_PINS[*]}"
$PIP install "${TF_PINS[@]}"

# ------------------------------------------------------------------ verify ---
say "Verifying"
"$VENV/bin/python" - <<'PY'
import sys
ok = True
def check(label, fn):
    global ok
    try: print(f"  {label:<12} {fn()}")
    except Exception as e:
        ok = False; print(f"  {label:<12} FAIL {type(e).__name__}: {str(e)[:120]}")
import numpy, pandas
check("numpy",  lambda: numpy.__version__)
check("pandas", lambda: pandas.__version__)
def _tf():
    import tensorflow as tf
    g = tf.config.list_physical_devices('GPU')
    if not g: raise RuntimeError("no GPU - tensorflow-metal not active")
    return f"{tf.__version__}  GPU:{len(g)}"
check("tensorflow", _tf)
def _mlx():
    import mlx.core as mx
    a = mx.ones((256,256)) @ mx.ones((256,256)); mx.eval(a)
    return f"OK  {mx.default_device()}"
check("mlx", _mlx)
check("pandas_ta", lambda: (__import__("pandas_ta"), "OK")[1])
sys.exit(0 if ok else 1)
PY
"$VENV/bin/freqtrade" --version | tail -2
say "Done. Activate with: source .venv/bin/activate"
