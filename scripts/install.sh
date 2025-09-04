#!/bin/bash
echo "virtualenv directory path: $VIRTUALENVS"
echo "submodlib library directory path: $SUBMODLIB"

uv venv $VIRTUALENVS/muvera_$HOSTNAME --python 3.10
source $VIRTUALENVS/muvera_$HOSTNAME/bin/activate

uv pip install -r muvera_requirements_py3_10.txt --index-strategy unsafe-best-match
uv pip install -e ./ColBERT
uv pip install -e $SUBMODLIB