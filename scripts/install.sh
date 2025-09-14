#!/bin/bash
echo "virtualenv directory path: $VIRTUALENVS"
echo "submodlib library directory path: $SUBMODLIB"

uv venv $VIRTUALENVS/muvera_$HOSTNAME --python 3.10
source $VIRTUALENVS/muvera_$HOSTNAME/bin/activate

uv pip install -r muvera_requirements_py3_10.txt --index-strategy unsafe-best-match

# If HOSTNAME is boa or gnu, then install from muvera_requirements_torch_gnu.txt
if [[ "$HOSTNAME" == "gnu" || "$HOSTNAME" == "boa" ]]
then
    uv pip install -r muvera_requirements_torch_gnu.txt --index-strategy unsafe-best-match
else
    uv pip install -r muvera_requirements_torch.txt --index-strategy unsafe-best-match
fi

uv pip install -e ./ColBERT
uv pip install -e $SUBMODLIB