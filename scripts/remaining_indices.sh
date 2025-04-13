CUDA_VISIBLE_DEVICES=1 \
    python3 -m src.colbert_embs \
        data.dataset_name=scifact \
        overwrite_index=True \
        index=True \
        augment=True \
        dbl_norm=True \
        method="augmented" \
        rh_num=4 &

CUDA_VISIBLE_DEVICES=2 \
    python3 -m src.colbert_embs \
        data.dataset_name=scifact \
        overwrite_index=True \
        index=True \
        augment=True \
        dbl_norm=True \
        method="augmented" \
        rh_num=5 &

CUDA_VISIBLE_DEVICES=3 \
    python3 -m src.colbert_embs \
        data.dataset_name=nfcorpus \
        overwrite_index=True \
        index=True \
        augment=True \
        dbl_norm=True \
        method="augmented" \
        rh_num=6 &


wait