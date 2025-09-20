#!/bin/bash
#SBATCH --partition=l40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --qos=l40
#SBATCH --gres=gpu:1             # Request 2 GPUs
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16        # Request 64 CPUs
#SBATCH --mem=300G                # Request 300 GB RAM
#SBATCH --output=logs/muvera-timing-%j.out  # Save stdout to logs/train-<jobid>.out
#SBATCH --error=logs/muvera-timing-%j.err   # Save stderr to logs/train-<jobid>.err

# Load modules if necessary
# module load python/3.10

tsec_log() {
  local outfile="$1"
  shift
  [[ ${1-} == -- ]] && shift

  # Require GNU time
  if ! [[ -x /usr/bin/time ]]; then
    echo "Error: GNU time (/usr/bin/time) not found." >&2
    return 127
  fi

  # Temp file to capture just the '%e' output from GNU time
  local tmp
  tmp=$(mktemp) || { echo "mktemp failed" >&2; return 1; }

  # Run the command normally; write elapsed seconds to $tmp
  # Force C locale so decimal uses '.' regardless of system locale.
  LC_ALL=C /usr/bin/time -f '%e' -o "$tmp" -- "$@"
  local status=$?

  # Read, trim, and clean up
  local secs
  secs=$(tr -d '\r\n' < "$tmp")
  rm -f "$tmp"

  # Print to stdout and append to the file
  printf '%s\n' "$secs"
  printf '%s\n' "$secs" >> "$outfile"

  return $status   # propagate the command's exit code
}

# Don't try to ping the internet, all the data is already on disk
export HF_HUB_OFFLINE=1

# Activate environment
source ~/muvera/bin/activate

cd ~/CMUVERA_IR_ref

echo "Arguments received: $1, $2, $3, $4, $5, $6, $7, $8"

# Run your command
# Submodlib
tsec_log ./timing_analysis_$1.txt -- env CUDA_VISIBLE_DEVICES=0 python3 -m src.endtoend k=1 method='sml' data.dataset_name=$1 embedder.mode="disk" submodlib.optimizer=naive embedder.mv_type='colbertv2-plaid' load_state=$3 submodlib.path_suffix=timing submodlib.stop_if_zero_gain=$5 submodlib.epsilon=$6 data.loader_type=$7 data.query_type=$8 submodlib.mega_q_batch_size=1

tsec_log ./timing_analysis_$1.txt -- env CUDA_VISIBLE_DEVICES=0 python3 -m src.endtoend k=1 method='sml' data.dataset_name=$1 embedder.mode="disk" submodlib.optimizer=lazy embedder.mv_type='colbertv2-plaid' load_state=$3 submodlib.path_suffix=timing submodlib.stop_if_zero_gain=$5 submodlib.epsilon=$6 data.loader_type=$7 data.query_type=$8 submodlib.mega_q_batch_size=1

tsec_log ./timing_analysis_$1.txt -- env CUDA_VISIBLE_DEVICES=0 python3 -m src.endtoend k=1 method='sml' data.dataset_name=$1 embedder.mode="disk" submodlib.optimizer=stoc embedder.mv_type='colbertv2-plaid' load_state=$3 submodlib.path_suffix=timing submodlib.stop_if_zero_gain=$5 submodlib.epsilon=0.5 data.loader_type=$7 data.query_type=$8 submodlib.mega_q_batch_size=1

tsec_log ./timing_analysis_$1.txt -- env CUDA_VISIBLE_DEVICES=0 python3 -m src.endtoend k=1 method='sml' data.dataset_name=$1 embedder.mode="disk" submodlib.optimizer=ltl embedder.mv_type='colbertv2-plaid' load_state=$3 submodlib.path_suffix=timing submodlib.stop_if_zero_gain=$5 submodlib.epsilon=0.1 data.loader_type=$7 data.query_type=$8 submodlib.mega_q_batch_size=1
tsec_log ./timing_analysis_$1.txt -- env CUDA_VISIBLE_DEVICES=0 python3 -m src.endtoend k=1 method='sml' data.dataset_name=$1 embedder.mode="disk" submodlib.optimizer=ltl embedder.mv_type='colbertv2-plaid' load_state=$3 submodlib.path_suffix=timing submodlib.stop_if_zero_gain=$5 submodlib.epsilon=0.5 data.loader_type=$7 data.query_type=$8 submodlib.mega_q_batch_size=1
tsec_log ./timing_analysis_$1.txt -- env CUDA_VISIBLE_DEVICES=0 python3 -m src.endtoend k=1 method='sml' data.dataset_name=$1 embedder.mode="disk" submodlib.optimizer=ltl embedder.mv_type='colbertv2-plaid' load_state=$3 submodlib.path_suffix=timing submodlib.stop_if_zero_gain=$5 submodlib.epsilon=0.9 data.loader_type=$7 data.query_type=$8 submodlib.mega_q_batch_size=1

# Exact greedy
tsec_log ./timing_analysis_$1.txt -- env CUDA_VISIBLE_DEVICES=0 python3 -m src.endtoend k=1 method='v0' embedder.mode="disk" baseline.distributed_search=False baseline.bucket_size=$1 data.dataset_name=$2 embedder.mv_type='colbertv2-plaid' load_state=$3 data.loader_type=$4 data.query_type=$5

# WARP iid
tsec_log ./timing_analysis_$1.txt -- env CUDA_VISIBLE_DEVICES=0 python3 -m src.xtr data.loader_type=$2 data.query_type=$3 data.dataset_name=$1 embedder.mode="disk" overwrite_index=False index=False augment=False method="xtr" embedder.type="BERT" embedder.mv_type="colbertv2-plaid" embedder.model="google/xtr-base-en" k=1

# Muvera iid
tsec_log ./timing_analysis_$1.txt -- env CUDA_VISIBLE_DEVICES=0 taskset -c 0-50 python3 -m src.colbert_embs k=1 data.loader_type=$2 data.query_type=$3 data.dataset_name=$1 overwrite_index=False index=False augment=False dbl_norm=True method="muvera_iid" muvera.num_repetitions=20 muvera.num_simhash_projections=5 muvera.projection_dimension=20 muvera.final_projection_dimension=2560 lin_dim=128 muvera.half_embs=False muvera.type="BERT" muvera.compress=False embedder.mode="disk"

# ColBERT iid
tsec_log ./timing_analysis_$1.txt -- env CUDA_VISIBLE_DEVICES=0 python3 -m src.colbert_embs k=1 embedder.mode="disk" augment=False method='baseline' data.loader_type=$2 data.query_type=$3 data.dataset_name=$1 index=False overwrite_index=False

# ColBERT angiogram
tsec_log ./timing_analysis_$1.txt -- env CUDA_VISIBLE_DEVICES=0 taskset -c 0-50 python3 -m src.colbert_embs k=1 method='augmented' num_rh_augment=8 data.dataset_name=$1 augment=True index=False dbl_norm=True embedder.mode='disk' colbert_topk=1 data.loader_type=$2 data.query_type=$3

tsec_log ./timing_analysis_$1.txt -- env CUDA_VISIBLE_DEVICES=0 taskset -c 0-50 python3 -m src.colbert_embs k=1 method='augmented' num_rh_augment=8 data.dataset_name=$1 augment=True index=False dbl_norm=True embedder.mode='disk' colbert_topk=10 data.loader_type=$2 data.query_type=$3

tsec_log ./timing_analysis_$1.txt -- env CUDA_VISIBLE_DEVICES=0 taskset -c 0-50 python3 -m src.colbert_embs k=1 method='augmented' num_rh_augment=8 data.dataset_name=$1 augment=True index=False dbl_norm=True embedder.mode='disk' colbert_topk=15 data.loader_type=$2 data.query_type=$3

tsec_log ./timing_analysis_$1.txt -- env CUDA_VISIBLE_DEVICES=0 taskset -c 0-50 python3 -m src.colbert_embs k=1 method='augmented' num_rh_augment=8 data.dataset_name=$1 augment=True index=False dbl_norm=True embedder.mode='disk' colbert_topk=20 data.loader_type=$2 data.query_type=$3

# ColBERT bypass
tsec_log ./timing_analysis_$1.txt -- env CUDA_VISIBLE_DEVICES=0 taskset -c 0-50 python3 -m src.colbert_embs k=1 method='internal' num_rh_augment=8 data.dataset_name=$1 augment=True index=False dbl_norm=True embedder.mode='disk' colbert_topk=1 data.loader_type=$2 data.query_type=$3 colbert_internal.rerank_internal=True colbert_internal.rerank_external=True

tsec_log ./timing_analysis_$1.txt -- env CUDA_VISIBLE_DEVICES=0 taskset -c 0-50 python3 -m src.colbert_embs k=1 method='internal' num_rh_augment=8 data.dataset_name=$1 augment=True index=False dbl_norm=True embedder.mode='disk' colbert_topk=10 data.loader_type=$2 data.query_type=$3 colbert_internal.rerank_internal=True colbert_internal.rerank_external=True

tsec_log ./timing_analysis_$1.txt -- env CUDA_VISIBLE_DEVICES=0 taskset -c 0-50 python3 -m src.colbert_embs k=1 method='internal' num_rh_augment=8 data.dataset_name=$1 augment=True index=False dbl_norm=True embedder.mode='disk' colbert_topk=15 data.loader_type=$2 data.query_type=$3 colbert_internal.rerank_internal=True colbert_internal.rerank_external=True

tsec_log ./timing_analysis_$1.txt -- env CUDA_VISIBLE_DEVICES=0 taskset -c 0-50 python3 -m src.colbert_embs k=1 method='internal' num_rh_augment=8 data.dataset_name=$1 augment=True index=False dbl_norm=True embedder.mode='disk' colbert_topk=20 data.loader_type=$2 data.query_type=$3 colbert_internal.rerank_internal=True colbert_internal.rerank_external=True