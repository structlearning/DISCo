#!/bin/bash
if [ -z "$1" ]; then
    echo "Error: Dataset name argument is required."
    exit 1
fi

### Exact GReedy followed by stochastic greedy

for bs in 0 50 100 200 400 800 1000; do
    python3 -m src.endtoend k=15 method='v0' baseline.bucket_size=$bs data.dataset_name=$1 embedder.mv_type='colbertv2-plaid'
done
