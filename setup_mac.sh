#!/bin/bash
# One-command setup for macOS:  bash setup_mac.sh
set -e
PY=$(command -v python3.11 || true)
if [ -z "$PY" ]; then
  echo "Python 3.11 not found. Install it with:  brew install python@3.11"
  exit 1
fi
$PY -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
[ -f .env ] || cp .env.example .env
echo ""
echo "Setup complete. Next:"
echo "  1) Open .env and paste your GROQ_API_KEY and TAVILY_API_KEY"
echo "  2) source .venv/bin/activate && streamlit run app.py"
