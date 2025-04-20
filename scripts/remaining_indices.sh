CUDA_VISIBLE_DEVICES=2 \
    python3 -m src.colbert_embs \
        data.dataset_name=scifact \
        overwrite_index=True \
        index=True \
        augment=True \
        dbl_norm=False \
        method="augmented" \
        rh_num=7 &
