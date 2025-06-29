#!/bin/bash
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <dataset_name>"
    exit 1
fi
dataset_name="$1"

# List of available GPUs
if [ "$HOSTNAME" = "elk" ]; then
    gpus=(1 2 3 4 6)
elif [ "$HOSTNAME" = "fox" ]; then
	# gpus=(0 1 2 3 4 )  # For FOX
	gpus=(1 2 3 4 )  # For FOX
elif [ "$HOSTNAME" = "dog" ]; then
    gpus=(0 1 2 3 4 5)
elif [ "$HOSTNAME" = "iitb-dgx2.iitb.ac.in" ]; then
    gpus=(0 1 2 3 4 5 6 7)
else
    echo "$HOSTNAME"
    exit
fi

num_gpus=${#gpus[@]}
counter=0

# for bs in 10 25 50 100 200; do
# for bs in 50 100 200; do
for bs in 100 200 ; do
    CUDA_VISIBLE_DEVICES=${gpus[$counter]} \
    python3 -m src.colbert_embs \
        k=15 \
        method='augmented' \
        num_rh_augment=8 \
        data.dataset_name=$dataset_name \
        augment=True \
        index=False \
        dbl_norm=True \
        embedder.mode='disk' \
        colbert_topk=$bs &
    counter=$((counter + 1))
    if [ "$counter" -eq "$num_gpus" ]; then
        wait
        counter=0
    fi
done
wait