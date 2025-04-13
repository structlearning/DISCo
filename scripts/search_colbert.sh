#!/bin/bash
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <dataset_name>"
    exit 1
fi
dataset_name="$1"

cuda_visible_devices=$(echo $CUDA_VISIBLE_DEVICES | tr ',' ' ')
num_gpus=$(echo $cuda_visible_devices | wc -w)
gpus=($cuda_visible_devices)


counter=0

for bs in 10 25 50 100 200; do

    CUDA_VISIBLE_DEVICES=${gpus[$counter]} \
    python3 -m src.colbert_embs \
        k=15 \
        method='augmented' \
        num_rh_augment=8 \
        data.dataset_name=$dataset_name \
        augment=True \
        index=False \
        colbert_topk=$bs &
    counter=$((counter + 1))
    if [ "$counter" -eq "$num_gpus" ]; then
        wait
        counter=0
    fi
done