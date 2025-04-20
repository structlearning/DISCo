#!/bin/bash

if [ "$HOSTNAME" = "elk" ]; then
    GPUS=(1 2 3 4 6)
elif [ "$HOSTNAME" = "fox" ]; then
    GPUS=(2 3 4 0 1)  # For FOX
else
    echo "$HOSTNAME"
    exit
fi

NUM_RH_AUGMENT=8
DATASET_NAME="$1"

for ((i = 0; i < NUM_RH_AUGMENT; i++)); do
    GPU_INDEX=$((i % ${#GPUS[@]}))
    GPU_ID=${GPUS[$GPU_INDEX]}
    
    echo "Launching rh_num=$i on GPU $GPU_ID"
    
    # Double augmentation flag on below
    CUDA_VISIBLE_DEVICES=$GPU_ID \
    python3 -m src.colbert_embs \
        data.dataset_name="$DATASET_NAME" \
        overwrite_index=False \
        index=True \
        augment=True \
        dbl_norm=False \
        method="augmented" \
        rh_num=$i &
    
done

wait
