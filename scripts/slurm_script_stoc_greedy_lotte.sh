#!/bin/bash
#SBATCH --partition=a40
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --qos=a40
#SBATCH --gres=gpu:2             # Request 2 GPUs
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16        # Request 64 CPUs
#SBATCH --mem=400G                # Request 200 GB RAM
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
# 10% of each dataset used as bucket size
CUDA_VISIBLE_DEVICES=0 python3 -m src.endtoend k=10 method='v0' embedder.mode="disk" baseline.distributed_search=False baseline.bucket_size=84300 data.dataset_name=pooled embedder.mv_type='colbertv2-plaid' load_state=$1 data.loader_type=$2 data.query_type=$3 &
CUDA_VISIBLE_DEVICES=1 python3 -m src.endtoend k=10 method='v0' embedder.mode="disk" baseline.distributed_search=False baseline.bucket_size=51200 data.dataset_name=science embedder.mv_type='colbertv2-plaid' load_state=$1 data.loader_type=$2 data.query_type=$3 &
# CUDA_VISIBLE_DEVICES=1 python3 -m src.endtoend k=10 method='v0' embedder.mode="disk" baseline.distributed_search=False baseline.bucket_size=19236 data.dataset_name=technology embedder.mv_type='colbertv2-plaid' load_state=$1 data.loader_type=$2 data.query_type=$3 &
# CUDA_VISIBLE_DEVICES=1 python3 -m src.endtoend k=10 method='v0' embedder.mode="disk" baseline.distributed_search=False baseline.bucket_size=6021 data.dataset_name=writing embedder.mv_type='colbertv2-plaid' load_state=$1 data.loader_type=$2 data.query_type=$3 &
wait