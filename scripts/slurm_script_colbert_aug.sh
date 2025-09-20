#!/bin/bash
# NOTE: If specifying l40, usually different jobs go on different nodes, so memory intensive jobs can survive.
# For a40, we have noticed that jobs sometimes go on the same node, so memory intensive jobs can crash.
# To handle this, may need to specify a --nodelist argument when submitting jobs to a40.
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --gpus=1             # Request 1 GPU
#SBATCH --cpus-per-task=32        # Request 64 CPUs
#SBATCH --mem=300G                # Request 300 GB RAM
#SBATCH --output=logs/muvera-%j.out  # Save stdout to logs/train-<jobid>.out
#SBATCH --error=logs/muvera-%j.err   # Save stderr to logs/train-<jobid>.err

# Load modules if necessary
# module load python/3.10
source ~/.bashrc

echo $(which g++)

echo "==== SLURM CONTEXT ===="
echo "JobID:         $SLURM_JOB_ID"
echo "ArrayTaskID:   ${SLURM_ARRAY_TASK_ID:-N/A}"
echo "Partition:     $SLURM_JOB_PARTITION"
echo "QOS:           ${SLURM_JOB_QOS:-N/A}"
echo "NodeList:      ${SLURM_NODELIST:-N/A}"
echo "NTasks:        ${SLURM_NTASKS:-N/A}  CPUs/Task: ${SLURM_CPUS_PER_TASK:-N/A}  GPUs: ${SLURM_GPUS:-N/A}"
echo "==============="

echo "Hostname: $(hostname)"
echo "Kernel:   $(uname -r)"
echo "GPU(s):"
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total --format=csv || true
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

# (Optional) summarize GPU family -> a40/l40/dgx
gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 2>/dev/null || echo "")
case "$gpu_name" in
  *A40*)  echo "GPU family detected: A40";;
  *L40*)  echo "GPU family detected: L40";;
  *A100*) echo "GPU family detected: DGX/A100";;
  *)      echo "GPU family detected: unknown ($gpu_name)";;
esac

# Don't try to ping the internet, all the data is already on disk
export HF_HUB_OFFLINE=1

# Activate environment
source ~/muvera/bin/activate

cd ~/CMUVERA_IR_ref

# Run your command
CUDA_VISIBLE_DEVICES=0 python3 -m src.colbert_embs k=10 method='augmented' num_rh_augment=8 data.dataset_name=$1 augment=True index=False dbl_norm=True embedder.mode='disk' colbert_topk=$2 data.loader_type=$3 data.query_type=$4