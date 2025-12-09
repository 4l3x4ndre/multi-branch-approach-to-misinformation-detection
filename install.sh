#!/bin/bash
set -e

# Check for poetry and install if not present
if ! command -v poetry &> /dev/null; then
    echo "Poetry not found. Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
    export PATH="$HOME/.local/bin:$PATH"
fi

# Install Python dependencies (creates/updates .venv)
poetry install

# Install spaCy model
poetry run python -m spacy download en_core_web_sm

echo "Installation complete."

