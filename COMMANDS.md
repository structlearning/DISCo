# Commands

In this file, we provide examples of all types of commands to run. Ensure that you have read README.md before using these. We demonstrate on the MSMarco and Pooled datasets.

PS: If required, you can manage GPU usage by putting CUDA_VISIBLE_DEVICES and CPU usage by putting `taskset -c min_cpu_id-max_cpu_id` before the command. It is advised to leverage GPUs if they are available.

## Indexing and Embedding Generation

We start with indexing commands. Indexing for PLAID automatically generates the necessary (Col)BERT embeddings that will be used by every method. The embeddings will be located at `./experiments/<dataset_name>/BERT/colbertv2-plaid/corpus/compressed_128`. The location of the indices varies.

Start by PLAID indexing, and then do everything else.

### PLAID

Index location: `./experiments/<dataset_name>/BERT/colbertv2-plaid/norm/indexes/nbits=2.noaug`

```
python3 -m src.colbert_embs \
    data.dataset_name=msmarco \
    overwrite_index=True \
    index=True \
    augment=False \
    loader_type=beir \
    query_type=forum \ # does not matter for beir
    method="baseline"

python3 -m src.colbert_embs \
    data.dataset_name=pooled \
    overwrite_index=True \
    index=True \
    augment=False \
    loader_type=lotte \
    query_type=forum \
    method="baseline"
```

### MUVERA

In the case of MUVERA, the fixed dimensional encodings are to be generated first for each dataset. They will be located at `./experiments/<dataset_name>/BERT/colbertv2-plaid/corpus/compressed_muvera_full_128`. Once this is done, the indexing commands can be run.

Index location: `./experiments/<dataset_name>/muvera_index_128.index`

Encoding commands -:

```
python3 -m src.cmuvera data.dataset_name="msmarco" data.loader_type=beir data.query_type=forum overwrite_index=False index=False augment=False dbl_norm=True method="muvera" muvera.num_repetitions=20 muvera.num_simhash_projections=5 muvera.projection_dimension=20 muvera.final_projection_dimension=2560 lin_dim=128 muvera.half_embs=False muvera.fresh_start=False muvera.type="BERT" muvera.compress=False embedder.mode="disk" muvera.parallel=True

python3 -m src.cmuvera data.dataset_name="pooled" data.loader_type=lotte data.query_type=forum overwrite_index=False index=False augment=False dbl_norm=True method="muvera" muvera.num_repetitions=20 muvera.num_simhash_projections=5 muvera.projection_dimension=20 muvera.final_projection_dimension=2560 lin_dim=128 muvera.half_embs=False muvera.fresh_start=False muvera.type="BERT" muvera.compress=False embedder.mode="disk" muvera.parallel=True
```

Indexing commands -:

```
python3 -m src.colbert_embs data.dataset_name="msmarco" data.loader_type=beir data.query_type=forum overwrite_index=True index=True augment=False dbl_norm=True method="muvera_iid" muvera.num_repetitions=20 muvera.num_simhash_projections=5 muvera.projection_dimension=20 muvera.final_projection_dimension=2560 lin_dim=128 muvera.half_embs=False muvera.type="BERT" muvera.compress=False embedder.mode="disk"

python3 -m src.colbert_embs data.dataset_name="pooled" data.loader_type=lotte data.query_type=forum overwrite_index=True index=True augment=False dbl_norm=True method="muvera_iid" muvera.num_repetitions=20 muvera.num_simhash_projections=5 muvera.projection_dimension=20 muvera.final_projection_dimension=2560 lin_dim=128 muvera.half_embs=False muvera.type="BERT" muvera.compress=False embedder.mode="disk"
```

### WARP

Note that WARP generates T5 embeddings of the corpus and indexes them. During search, it embeds the queries using T5. Re-scoring for coverage happens using our BERT embeddings however.

Index location: `./experiments/<dataset_name>/BERT/colbertv2-plaid/norm/indexes/xtr_nbits=2.noaug_xtr-base-en`

```
python3 -m src.xtr data.dataset_name=msmarco data.loader_type=beir data.query_type=forum embedder.mode="disk" overwrite_index=True index=True augment=False method="xtr" embedder.type="WARP" embedder.mv_type="xtr-base"

python3 -m src.xtr data.dataset_name=pooled data.loader_type=lotte data.query_type=forum embedder.mode="disk" overwrite_index=True index=True augment=False method="xtr" embedder.type="WARP" embedder.mv_type="xtr-base"
```

### DISCo

DISCo generates one index per random hyperplane out of the R hyperplanes that are sampled. By default, R = 8.

```
NUM_RH_AUGMENT=8
GPUS=(0 1 2 3 4 5 6 7) # example
LOADER_TYPE=beir
QUERY_TYPE=forum
DATASET_NAME=msmarco

for ((i = 0; i < NUM_RH_AUGMENT; i++)); do
    GPU_INDEX=$((i % ${#GPUS[@]}))
    GPU_ID=${GPUS[$GPU_INDEX]}

    echo "Launching rh_num=$i on GPU $GPU_ID"

    # double augmentation flag on below

    # Do not separate line by line using slash even though it looks aesthetic.
    # rh_num does not get set if that is done.
    CUDA_VISIBLE_DEVICES=$GPU_ID \
    python3 -m src.colbert_embs data.loader_type=$LOADER data.query_type=$QUERY data.dataset_name=$DATASET_NAME embedder.mode="disk" overwrite_index=True index=True augment=True dbl_norm=True method="augmented" generate_new_rh=True rh_num=$i &
```

The above script can easily be modified for LoTTE datasets.

### Running src/calculate_docid_to_batch_info.py

After PLAID indexing has been run and the BERT embeddings are available, we need to run src/calculate_docid_to_batch_info.py to ensure that our disk mode style access works.

`python src/calculate_docid_to_batch_info.py <dataset_name>`

## Running index style retrievers and DISCo

One may set `overwrite_index` and `index` to False in each of the above indexing commands, and add `k=10`, to perform search over the index for a subset of size 10.

Specifically in the case of DISCo, one also needs to set `method` to `internal`. For the late pooling ablation, one may use `method=augmented`. Here is an example for DISCo and the late pooling ablation on the Pooled dataset.

```
python3 -m src.colbert_embs k=10 method='internal' num_rh_augment=8 data.dataset_name=pooled augment=True index=False dbl_norm=True embedder.mode='disk' colbert_topk=1 data.loader_type=lotte data.query_type=forum colbert_internal.rerank_internal=True colbert_internal.rerank_external=True

python3 -m src.colbert_embs k=10 method='augmented' num_rh_augment=8 data.dataset_name=pooled augment=True index=False dbl_norm=True embedder.mode='disk' colbert_topk=1 data.loader_type=lotte data.query_type=forum
```

## Running exact greedy and submodlib solver commands

These commands also require the existence of aforementioned the BERT embeddings. Here is an example of running multiple variations of greedy on the MSMarco dataset.

```
LOADER=beir
QTYPE=forum
DATASET=msmarco

common_base=(
  "embedder.mode=disk"
  "embedder.mv_type=colbertv2-plaid"
  "data.dataset_name=$DATASET"
  "data.loader_type=$LOADER"
  "data.query_type=$QTYPE"
)

endtoend_common=(
  "k=10"
  "submodlib.mega_q_batch_size=100"
  "submodlib.stop_if_zero_gain=False"
)

# Submodlib methods
python3 -m src.endtoend "method=sml" "${common_base[@]}" "${endtoend_common[@]}" "submodlib.optimizer=lazy"

python -m src.endtoend "method=sml" "${common_base[@]}" "${endtoend_common[@]}" "submodlib.optimizer=stoc" "submodlib.epsilon=0.5" path_suffix=submodlib_no_stop_eps0.5


python3 -m src.endtoend "method=sml" "${common_base[@]}" "${endtoend_common[@]}" "submodlib.optimizer=ltl"  "submodlib.epsilon=0.1"

python3 -m src.endtoend "method=sml" "${common_base[@]}" "${endtoend_common[@]}" "submodlib.optimizer=ltl"  "submodlib.epsilon=0.5" path_suffix=submodlib_no_stop_eps0.5

python3 -m src.endtoend "method=sml" "${common_base[@]}" "${endtoend_common[@]}" "submodlib.optimizer=ltl"  "submodlib.epsilon=0.9" path_suffix=submodlib_no_stop_eps0.9

# Exact greedy
python3 -m src.endtoend "method=v0" "${common_base[@]}" "baseline.distributed_search=False" "baseline.bucket_size=$BUCKET" k=10
```