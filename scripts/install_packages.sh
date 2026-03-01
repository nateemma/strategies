#!/bin/zsh

# script to install additional packages for the strategies in this repo

# I find this useful in that I normally have to do this if I upgrade something fundamental like python or anaconda.
# Also, I sometimes get in a mess with package dependencies and have to reset
# Either of these requires me to do a full install of freqtrade (sh setup.sh -r), and then re-install these packages

# Notes:
# - if you need conda versions of packages, you'll need to install those from within the freqtrade venv
# - at the current time, conda packages for the M1 Mac are a mess of conflicting version requirements, so beware
# - the neural network-based strategies require either keras & tensorflow, or darts & pytorch

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

# install generally used packages
pkg_general=("finta" "prettytable" "PyWavelets" "simdkalman" "pykalman" "scipy" "scikit-learn" \
"ast_comments" "rich" "xgboost" "lightgbm" "statsmodels" "imblearn" "tensorflow" "keras" "pandas-ta")

if [[ $(prompt_user "Install general packages?: ") -eq 1 ]]; then
  echo ""
  for pkg in $pkg_general; do
    pip3 install $pkg
  done

  # Force numpy < 2.0 to avoid the "umath failed to import" error
  # This is critical for compatibility with TensorFlow 2.16 and other legacy compiled modules.
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
      echo "Installing TensorFlow for Apple Silicon..."
      pip3 install --upgrade tensorflow-macos
      pip3 install --upgrade tensorflow-metal
      pip3 install --upgrade keras
      pip3 install --upgrade pandas
  else
    echo "Installing standard TensorFlow..."
    pip3 install --upgrade tensorflow
    pip3 install --upgrade keras
    pip3 install --upgrade pandas
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