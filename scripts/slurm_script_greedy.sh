#!/bin/bash
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --gpus=1             # Request 1 GPU
#SBATCH --cpus-per-task=32        # Request 64 CPUs
#SBATCH --mem=400G                # Request 500 GB RAM
#SBATCH --output=logs/muvera-%j.out  # Save stdout to logs/train-<jobid>.out
#SBATCH --error=logs/muvera-%j.err   # Save stderr to logs/train-<jobid>.err

# Load modules if necessary
# module load python/3.10

# Don't try to ping the internet, all the data is already on disk
export HF_HUB_OFFLINE=1

# Activate environment
source ~/muvera/bin/activate

cd ~/CMUVERA_IR_ref

# Run your command
CUDA_VISIBLE_DEVICES=0 python3 -m src.endtoend k=10 method='v0' embedder.mode="disk" baseline.distributed_search=False baseline.bucket_size=$1 data.dataset_name=$2 embedder.mv_type='colbertv2-plaid' load_state=$3 data.loader_type=$4 data.query_type=$5