#!/bin/bash
#SBATCH --partition=l40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --qos=l40
#SBATCH --gres=gpu:1             # Request 2 GPUs
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16        # Request 64 CPUs
#SBATCH --mem=350G                # Request 350 GB RAM
#SBATCH --output=logs/muvera-timing-%j.out  # Save stdout to logs/train-<jobid>.out
#SBATCH --error=logs/muvera-timing-%j.err   # Save stderr to logs/train-<jobid>.err

# Don't try to ping the internet, all the data is already on disk
export HF_HUB_OFFLINE=1

# Activate environment
source ~/muvera/bin/activate

cd ~/CMUVERA_IR_ref

# --- timing helper (GNU /usr/bin/time) ---
# Usage:
#   secs=$(tsec_log_lbl timing_analysis_${DATASET}.txt "sml_naive" -- your command ...)
#   echo "That took $secs seconds"

tsec_log() {
  local outfile="$1"
  local label="$2"
  shift 2
  [[ ${1-} == -- ]] && shift

  # Capture only the elapsed seconds
  local tmp
  tmp=$(mktemp) || { echo "mktemp failed" >&2; return 1; }
  LC_ALL=C time -f '%e' -o "$tmp" -- "$@"
  local status=$?

  # Read seconds, clean up
  local secs
  secs=$(tr -d '\r\n' < "$tmp")
  rm -f "$tmp"

  # Print seconds to stdout
  printf '%s\n' "$secs"

  # Ensure CSV header once
  if [[ ! -f "$outfile" || ! -s "$outfile" ]]; then
    printf 'name,seconds\n' >> "$outfile"
  fi

  # CSV-escape label (wrap in quotes, double any internal quotes)
  local esc_label=${label//\"/\"\"}
  printf '"%s",%s\n' "$esc_label" "$secs" >> "$outfile"

  return $status   # propagate the command's exit code
}


# --- parse long options (GNU getopt) ---
PARSED=$(getopt -o d:l:L:q:b:o:c:n: \
  --long dataset:,load_state:,loader_type:,query_type:,bucket_size:,outfile:,cuda:,num_queries: \
  -- "$@") || { echo "Bad args"; exit 2; }
eval set -- "$PARSED"

DATASET= ; LOAD_STATE= ; LOADER= ; QTYPE= ; BUCKET=
OUTFILE= ; CUDA=0; NUM_QUERIES= ;
while true; do
  case "$1" in
    -d|--dataset)      DATASET="$2"; shift 2;;
    -l|--load_state)   LOAD_STATE="$2"; shift 2;;
    -L|--loader_type)  LOADER="$2";   shift 2;;
    -q|--query_type)   QTYPE="$2";    shift 2;;
    -b|--bucket_size)  BUCKET="$2";   shift 2;;
    -o|--outfile)      OUTFILE="$2";  shift 2;;
    -c|--cuda)         CUDA="$2";     shift 2;;
    -n|--num_queries)  NUM_QUERIES="$2";     shift 2;;
    --) shift; break;;
    *) echo "Internal parsing error"; exit 3;;
  esac
done

# Requireds
: "${DATASET:?--dataset is required}"
: "${LOADER:?--loader_type is required}"
: "${QTYPE:?--query_type is required}"
OUTFILE="${OUTFILE:-./timing_analysis_${DATASET}.txt}"
NUM_QUERIES="${NUM_QUERIES:-100}"

echo "Args: dataset=$DATASET load_state=${LOAD_STATE:-<none>} loader=$LOADER query=$QTYPE bucket=${BUCKET:-<none>} cuda=$CUDA"
echo "Logging times to: $OUTFILE"
echo "Number of queries for timing: $NUM_QUERIES"

# --- common arg arrays (order-safe, no word splitting) ---
common_base=(
  "embedder.mode=disk"
  "embedder.mv_type=colbertv2-plaid"
  "embedder.num_queries=$NUM_QUERIES"
  "data.dataset_name=$DATASET"
  "data.loader_type=$LOADER"
  "data.query_type=$QTYPE"
)

endtoend_common=(
  "k=1"
  "submodlib.path_suffix=timing"
  "submodlib.mega_q_batch_size=1"
)
[[ -n "$LOAD_STATE" ]] && endtoend_common+=("load_state=$LOAD_STATE")

# WARP iid (xtr)
tsec_log "$OUTFILE" "WARP iid" -- env CUDA_VISIBLE_DEVICES="$CUDA" \
  python3 -m src.xtr "k=1" "${common_base[@]}" \
  "data.dataset_name=$DATASET" "data.loader_type=$LOADER" "data.query_type=$QTYPE" \
  "overwrite_index=False" "index=False" "augment=False" "method=xtr" \
  "embedder.type=BERT" "embedder.mv_type=colbertv2-plaid" "embedder.model=google/xtr-base-en"

# Muvera iid
tsec_log "$OUTFILE" "MUVERA iid" -- env CUDA_VISIBLE_DEVICES="$CUDA" taskset -c 0-50 \
  python3 -m src.colbert_embs "k=1" "${common_base[@]}" \
  "overwrite_index=False" "index=False" "augment=False" "dbl_norm=True" "method=muvera_iid" \
  "muvera.num_repetitions=20" "muvera.num_simhash_projections=5" \
  "muvera.projection_dimension=20" "muvera.final_projection_dimension=2560" "lin_dim=128" \
  "muvera.half_embs=False" "muvera.type=BERT" "muvera.compress=False"

# ColBERT iid
tsec_log "$OUTFILE" "ColBERT iid" -- env CUDA_VISIBLE_DEVICES="$CUDA" \
  python3 -m src.colbert_embs "k=1" "${common_base[@]}" \
  "augment=False" "method=baseline" "index=False" "overwrite_index=False"

# ColBERT angiogram (augmented)
for topk in 1 10 15 20; do
  tsec_log "$OUTFILE" "ColBERT angiogram - $topk" -- env CUDA_VISIBLE_DEVICES="$CUDA" taskset -c 0-50 \
    python3 -m src.colbert_embs "${common_base[@]}" "k=1" "method=augmented" "num_rh_augment=8" \
    "data.dataset_name=$DATASET" "augment=True" "index=False" "dbl_norm=True" \
    "embedder.mode=disk" "colbert_topk=$topk" "data.loader_type=$LOADER" "data.query_type=$QTYPE"
done

# ColBERT bypass (internal)
for topk in 1 10 15; do
  tsec_log "$OUTFILE" "ColBERT bypass - $topk" -- env CUDA_VISIBLE_DEVICES="$CUDA" taskset -c 0-50 \
    python3 -m src.colbert_embs "${common_base[@]}" "k=1" "method=internal" "num_rh_augment=8" \
    "data.dataset_name=$DATASET" "augment=True" "index=False" "dbl_norm=True" \
    "embedder.mode=disk" "colbert_topk=$topk" "data.loader_type=$LOADER" "data.query_type=$QTYPE" \
    "colbert_internal.rerank_internal=True" "colbert_internal.rerank_external=True"
done
