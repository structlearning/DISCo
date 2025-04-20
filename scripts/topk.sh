python3 -m src.colbert_embs \
        k=15 \
        method='baseline' \
        data.dataset_name=nfcorpus \
        augment=False \
        index=False &
        
python3 -m src.colbert_embs \
        k=15 \
        method='baseline' \
        data.dataset_name=scifact \
        augment=False \
        index=False        