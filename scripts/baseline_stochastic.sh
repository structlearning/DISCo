#!/bin/bash
if [ -z "$1" ]; then
    echo "Error: Dataset name argument is required."
    exit 1
fi

# List of available GPUs
if [ "$HOSTNAME" = "elk" ]; then
    GPUS=(1 2 3 4 6)
elif [ "$HOSTNAME" = "fox" ]; then
	GPUS=(0 1 2 3 4 )  # For FOX
elif [ "$HOSTNAME" = "dog" ]; then
    GPUS=(0 1 2 3 4 5)
elif [ "$HOSTNAME" = "iitb-dgx2.iitb.ac.in" ]; then
    GPUS=(0 1 2 3 4 5 6 7)
else
    echo "$HOSTNAME"
    exit
fi

### Exact GReedy followed by stochastic greedy

i=0

for bs in 0 50 100 200 400 800 1000; do
    GPU_ID=${GPUS[$i]}
    CUDA_VISIBLE_DEVICES=$GPU_ID python3 -m src.endtoend k=15 method='v0' baseline.bucket_size=$bs data.dataset_name=$1 embedder.mv_type='colbertv2-plaid' &
    i=$((i + 1))
    if [ $((i % ${#GPUS[@]})) -eq 0 ];
    then
        wait
        i=0
    fi
done
