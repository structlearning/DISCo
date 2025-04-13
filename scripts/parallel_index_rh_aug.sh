#!/bin/bash

# List of available GPUs
GPUS=(0 1 2 3 4 )  # For FOX
NUM_RH_AUGMENT=8  
DATASET_NAME="scifact"

for ((i=0; i<NUM_RH_AUGMENT; i++)); do
    GPU_INDEX=$((i % ${#GPUS[@]}))
    GPU_ID=${GPUS[$GPU_INDEX]}
    
    echo "Launching rh_num=$i on GPU $GPU_ID"
    
    # double augmentation flag on below

    CUDA_VISIBLE_DEVICES=$GPU_ID \
    python3 -m src.colbert_embs \
        data.dataset_name=$DATASET_NAME \
        overwrite_index=False \
        index=True \
        augment=True \
        dbl_norm=True \
        method="augmented" \ 
        generate_new_rh=True \
        rh_num=$i &

    # double augmentation flag off below

    # CUDA_VISIBLE_DEVICES=$GPU_ID \
    # python3 -m src.colbert_embs \
    #     data.dataset_name=$DATASET_NAME \
    #     overwrite_index=False \
    #     index=True \
    #     augment=True \
    #     dbl_norm=False \
    #     method="augmented" \
    #     generate_new_rh=True \
    #     rh_num=$i &

done

