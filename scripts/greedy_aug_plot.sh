#!/bin/bash
# List of available GPUs
if [ "$HOSTNAME" = "elk" ]; then
    GPUS=(1 2 3 4 6)
elif [ "$HOSTNAME" = "fox" ]; then
	GPUS=(0 1 2 3 4 )  # For FOX
elif [ "$HOSTNAME" = "dog" ]; then
    GPUS=(0 1 2 3 4 5)
elif [ "$HOSTNAME" = "bee" ]; then
    GPUS=(0 1 2 3)
elif [ "$HOSTNAME" = "iitb-dgx2.iitb.ac.in" ]; then
    GPUS=(0 1 2 3 4 5 6 7)
else
    echo "$HOSTNAME"
    exit
fi

NUM_RH_AUGMENT=8
DATASET_NAME="$1"
LOADER="$2"
QUERY="$3"

# Iterate over number of hyperplanes sampled
for ((num_hyperplanes=1; num_hyperplanes<=NUM_RH_AUGMENT; num_hyperplanes++)); do
    echo "Running with num_hyperplanes=$num_hyperplanes"
    GPU_INDEX=$((num_hyperplanes % ${#GPUS[@]}))
    GPU_ID=${GPUS[$GPU_INDEX]}
    CUDA_VISIBLE_DEVICES=$GPU_ID python3 -m src.endtoend k=10 method='augmented' embedder.mode="disk" baseline.distributed_search=False baseline.bucket_size=0 data.dataset_name=$DATASET_NAME data.loader_type=$LOADER data.query_type=$QUERY embedder.mv_type='colbertv2-plaid' load_state=False dbl_norm=False num_rh_augment=$num_hyperplanes &
done

wait