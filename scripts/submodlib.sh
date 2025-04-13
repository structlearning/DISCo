#!/bin/bash
if [ -z "$1" ]; then
    echo "Error: Dataset name argument is required."
    exit 1
fi

for method in 'lazy' 'ltl' 'naive'; do
    python3 -m src.endtoend k=15 method='sml' data.dataset_name=$1 submodlib.optimizer=$method embedder.mv_type='colbertv2-plaid'
done