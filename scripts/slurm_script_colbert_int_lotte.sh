#!/bin/bash
#SBATCH --partition=l40
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --qos=l40
#SBATCH --gres=gpu:2             # Request 2 GPUs
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=64        # Request 64 CPUs
#SBATCH --mem=250G                # Request 200 GB RAM
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
CUDA_VISIBLE_DEVICES=0 python3 -m src.colbert_embs k=10 embedder.mode="disk" colbert_internal.rerank_internal=True colbert_internal.rerank_external=True data.dataset_name=$1 augment=True index=False dbl_norm=True colbert_topk=1 data.loader_type=$2 data.query_type=$3 &
CUDA_VISIBLE_DEVICES=0 python3 -m src.colbert_embs k=10 embedder.mode="disk" colbert_internal.rerank_internal=True colbert_internal.rerank_external=True data.dataset_name=$1 augment=True index=False dbl_norm=True colbert_topk=10 data.loader_type=$2 data.query_type=$3 &
CUDA_VISIBLE_DEVICES=1 python3 -m src.colbert_embs k=10 embedder.mode="disk" colbert_internal.rerank_internal=True colbert_internal.rerank_external=True data.dataset_name=$1 augment=True index=False dbl_norm=True colbert_topk=15 data.loader_type=$2 data.query_type=$3 &
CUDA_VISIBLE_DEVICES=1 python3 -m src.colbert_embs k=10 embedder.mode="disk" colbert_internal.rerank_internal=True colbert_internal.rerank_external=True data.dataset_name=$1 augment=True index=False dbl_norm=True colbert_topk=20 data.loader_type=$2 data.query_type=$3 &
wait