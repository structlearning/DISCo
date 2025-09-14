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

python3 -m src.colbert_embs \
    data.dataset_name="$DATASET_NAME" \
    data.loader_type=$LOADER \
    data.query_type=$QUERY \
    embedder.mode="disk" \
    overwrite_index=True \
    index=True \
    augment=False \
    method="baseline" &

for ((i = 0; i < NUM_RH_AUGMENT; i++)); do
    GPU_INDEX=$((i % ${#GPUS[@]}))
    GPU_ID=${GPUS[$GPU_INDEX]}

    echo "Launching rh_num=$i on GPU $GPU_ID"

    # double augmentation flag on below

    # Do not separate line by line using slash even though it looks aesthetic.
    # rh_num does not get set if that is done.
    CUDA_VISIBLE_DEVICES=$GPU_ID \
    python3 -m src.colbert_embs data.loader_type=$LOADER data.query_type=$QUERY data.dataset_name=$DATASET_NAME embedder.mode="disk" overwrite_index=True index=True augment=True dbl_norm=True method="augmented" generate_new_rh=True rh_num=$i &

    # Double augmentation flag off below
    # CUDA_VISIBLE_DEVICES=$GPU_ID \
    # python3 -m src.colbert_embs \
    #     data.dataset_name="$DATASET_NAME" \
    #     overwrite_index=False \
    #     index=True \
    #     augment=True \
    #     dbl_norm=False \
    #     method="augmented" \
    #     generate_new_rh=True \
    #     rh_num=$i &
done
wait
