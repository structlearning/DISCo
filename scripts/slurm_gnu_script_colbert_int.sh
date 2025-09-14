#!/bin/bash

#SBATCH --partition=Standard
#SBATCH --gpus=7g.180gb:1             # Request 2 45GB GPUs
#SBATCH --cpus-per-task=64        # Request 64 CPUs
#SBATCH --mem=400G                # Request 200 GB RAM
#SBATCH --output=logs/muvera-%j.out  # Save stdout to logs/train-<jobid>.out
#SBATCH --error=logs/muvera-%j.err   # Save stderr to logs/train-<jobid>.err

# Load modules if necessary
# module load python/3.10
source /mnt/nas/pritish/dotfiles/.bashrc

# Don't try to ping the internet, all the data is already on disk
# export HF_HUB_OFFLINE=1

# Activate environment
source $INF_PATH/virtualenvs/muvera_gnu/bin/activate

cd $DGX_PATH/CMUVERA_IR_ref

# Run your command
# CUDA_VISIBLE_DEVICES=0 python3 -m src.colbert_embs k=10 embedder.mode="disk" colbert_internal.rerank_internal=True colbert_internal.rerank_external=True data.dataset_name=$1 augment=True index=False dbl_norm=True colbert_topk=1 data.loader_type=$2 data.query_type=$3 &
# CUDA_VISIBLE_DEVICES=0 python3 -m src.colbert_embs k=10 embedder.mode="disk" colbert_internal.rerank_internal=True colbert_internal.rerank_external=True data.dataset_name=$1 augment=True index=False dbl_norm=True colbert_topk=10 data.loader_type=$2 data.query_type=$3 &
# CUDA_VISIBLE_DEVICES=0 python3 -m src.colbert_embs k=10 embedder.mode="disk" colbert_internal.rerank_internal=True colbert_internal.rerank_external=True data.dataset_name=$1 augment=True index=False dbl_norm=True colbert_topk=15 data.loader_type=$2 data.query_type=$3 &
CUDA_VISIBLE_DEVICES=0 python3 -m src.colbert_embs k=10 embedder.mode="disk" colbert_internal.rerank_internal=True colbert_internal.rerank_external=True data.dataset_name=$1 augment=True index=False dbl_norm=True colbert_topk=20 data.loader_type=$2 data.query_type=$3 &
wait