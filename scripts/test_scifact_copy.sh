
python3 -m src.endtoend embedder.emb_dim=20 data.dataset_name='scifact' muvera.corpus_batch_size=10240 muvera.compress_dim=20 muvera.lsh.hash_dim=8 embedder.mode='mem' retriever.mode='disk' retriever.muvera_bucket_size=10 corpus_batch_size=10240;
for bs in 10 50 100 200 500; do
    python3 -m src.endtoend embedder.emb_dim=20 data.dataset_name='scifact' muvera.corpus_batch_size=10240 muvera.compress_dim=20 muvera.lsh.hash_dim=8 embedder.mode='mem' retriever.mode='disk' retriever.muvera_bucket_size=$bs corpus_batch_size=10240;
done

for bs in 250 500 1000 2500 5183; do
    python3 -m src.endtoend embedder.emb_dim=20 data.dataset_name='scifact' embedder.mode='mem' retriever.mode='disk' baseline.bucket_size=$bs corpus_batch_size=10240 method='v0';
done


python3 -m src.endtoend embedder.emb_dim=20 data.dataset_name='scifact' muvera.corpus_batch_size=10240 muvera.compress_dim=20 muvera.lsh.hash_dim=9 embedder.mode='mem' retriever.mode='disk' retriever.muvera_bucket_size=10 corpus_batch_size=10240;
for bs in 10 50 100 200 500; do
    python3 -m src.endtoend embedder.emb_dim=20 data.dataset_name='scifact' muvera.corpus_batch_size=10240 muvera.compress_dim=20 muvera.lsh.hash_dim=9 embedder.mode='mem' retriever.mode='disk' retriever.muvera_bucket_size=$bs corpus_batch_size=10240;
done


python3 -m src.endtoend embedder.emb_dim=20 data.dataset_name='scifact' muvera.corpus_batch_size=10240 muvera.compress_dim=10 muvera.lsh.hash_dim=10 embedder.mode='mem' retriever.mode='disk' retriever.muvera_bucket_size=10 corpus_batch_size=10240;

for bs in 10 50 100 200 500; do
    python3 -m src.endtoend embedder.emb_dim=20 data.dataset_name='scifact' muvera.corpus_batch_size=10240 muvera.compress_dim=10 muvera.lsh.hash_dim=10 embedder.mode='mem' retriever.mode='disk' retriever.muvera_bucket_size=$bs corpus_batch_size=10240;
done