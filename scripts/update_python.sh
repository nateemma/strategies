#!/bin/zsh
# One-time setup script for Python 3.11 on macOS

set -e  # stop if any command fails

echo "🔹 Step 1: Updating Homebrew..."
brew update

echo "🔹 Step 2: Installing Python 3.11..."
brew install python@3.11 || brew upgrade python@3.11

echo "🔹 Step 3: Linking Python 3.11 as default..."
brew link python@3.11 --force --overwrite


pyenv install 3.11

# Find Homebrew path (Intel vs Apple Silicon)
BREW_PREFIX=$(brew --prefix)
PYTHON_PATH="$BREW_PREFIX/bin/python3.11"
PIP_PATH="$BREW_PREFIX/bin/pip3.11"

echo "🔹 Step 4: Adding to shell profile..."
PROFILE="$HOME/.zshrc"
if ! grep -q "$BREW_PREFIX/bin" "$PROFILE"; then
    echo "export PATH=\"$BREW_PREFIX/bin:\$PATH\"" >> "$PROFILE"
    echo "✅ Added Homebrew Python to PATH in $PROFILE"
fi

echo "🔹 Step 5: Creating venvs directory..."
mkdir -p "$HOME/venvs"

echo "🔹 Step 6: Testing Python installation..."
$PYTHON_PATH --version
$PIP_PATH --version

echo "🔹 Step 7: Creating a test virtual environment..."
$PYTHON_PATH -m venv "$HOME/venvs/test_py313"
source "$HOME/venvs/test_py313/bin/activate"
python --version
pip --version
deactivate

echo "✅ Setup complete!"
echo "👉 Restart your terminal or run: source $PROFILE"
echo "👉 Create new venvs with: python3.11 -m venv ~/venvs/myproject"
echo "👉 In IDEs, select: ~/venvs/myproject/bin/python"
