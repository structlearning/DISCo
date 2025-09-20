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

source ~/.bashrc
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
  printf '"%s",%s (k=5)\n' "$esc_label" "$secs" >> "$outfile"

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
OUTFILE="${OUTFILE:-./timing_analysis_submodlib_${DATASET}.txt}"
NUM_QUERIES="${NUM_QUERIES:-1}"

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
  "k=5"
  "submodlib.path_suffix=timing"
  "submodlib.mega_q_batch_size=1"
)
[[ -n "$LOAD_STATE" ]] && endtoend_common+=("load_state=$LOAD_STATE")

# ---------- Run your series ----------
# Submodlib (sml)
tsec_log "$OUTFILE" "submodlib naive" -- env CUDA_VISIBLE_DEVICES="$CUDA" \
  python3 -m src.endtoend "method=sml" "${common_base[@]}" "${endtoend_common[@]}" "submodlib.optimizer=naive"

tsec_log "$OUTFILE" "submodlib lazy" -- env CUDA_VISIBLE_DEVICES="$CUDA" \
  python3 -m src.endtoend "method=sml" "${common_base[@]}" "${endtoend_common[@]}" "submodlib.optimizer=lazy"

tsec_log "$OUTFILE" "submodlib stochastic 0.5" -- env CUDA_VISIBLE_DEVICES="$CUDA" \
  python3 -m src.endtoend "method=sml" "${common_base[@]}" "${endtoend_common[@]}" "submodlib.optimizer=stoc" "submodlib.epsilon=0.5"

tsec_log "$OUTFILE" "submodlib ltl 0.1" -- env CUDA_VISIBLE_DEVICES="$CUDA" \
  python3 -m src.endtoend "method=sml" "${common_base[@]}" "${endtoend_common[@]}" "submodlib.optimizer=ltl"  "submodlib.epsilon=0.1"
tsec_log "$OUTFILE" "submodlib ltl 0.5" -- env CUDA_VISIBLE_DEVICES="$CUDA" \
  python3 -m src.endtoend "method=sml" "${common_base[@]}" "${endtoend_common[@]}" "submodlib.optimizer=ltl"  "submodlib.epsilon=0.5"
tsec_log "$OUTFILE" "submodlib ltl 0.9" -- env CUDA_VISIBLE_DEVICES="$CUDA" \
  python3 -m src.endtoend "method=sml" "${common_base[@]}" "${endtoend_common[@]}" "submodlib.optimizer=ltl"  "submodlib.epsilon=0.9"

# ---------- Run your series ----------
Exact greedy (v0 baseline) — needs bucket size
if [[ -n "$BUCKET" ]]; then
  tsec_log "$OUTFILE" "exact greedy" -- env CUDA_VISIBLE_DEVICES="$CUDA" \
    python3 -m src.endtoend "method=v0" "${common_base[@]}" \
    "baseline.distributed_search=False" "baseline.bucket_size=$BUCKET" k=1
else
  echo "Skipping exact greedy: --bucket_size not provided" >&2
fi