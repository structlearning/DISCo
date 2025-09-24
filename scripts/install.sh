#!/bin/bash
# VIRTUALENVS is the directory where virtualenvs are stored.
# SUBMODLIB points to where you have cloned the submodlib repository.
echo "virtualenv directory path: $VIRTUALENVS"
echo "submodlib library directory path: $SUBMODLIB"

# Set your virtualenv directory name here
VENV_NAME="muvera"
echo "virtualenv name: $VENV_NAME"

# We use uv as our package manager.
uv venv $VIRTUALENVS/$VENV_NAME --python 3.10
source $VIRTUALENVS/$VENV_NAME/bin/activate

uv pip install -r muvera_requirements_py3_10.txt --index-strategy unsafe-best-match
uv pip install -r muvera_requirements_torch.txt --index-strategy unsafe-best-match

# Install from ColBERT and submodlib directories.
# Note that in the case of WARP, we only need to clone it .
uv pip install -e ./ColBERT
uv pip install -e $SUBMODLIB