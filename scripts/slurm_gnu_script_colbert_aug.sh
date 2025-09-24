#!/bin/bash
#SBATCH --job-name=colbert_augmented
#SBATCH --partition=Standard
#SBATCH --gpus=2g.45gb:2             # Request 2 45GB GPU
#SBATCH --cpus-per-task=32        # Request 64 CPUs
#SBATCH --mem=1T                # Request 1 TB RAM
#SBATCH --time=144:00:00             # Set a time limit of 6 days
#SBATCH --output=logs/muvera_%j.out  # Save stdout to logs/train-<jobid>.out
#SBATCH --error=logs/muvera_%j.err   # Save stderr to logs/train-<jobid>.err

# Load modules if necessary
# module load python/3.10
source /mnt/nas/pritish/dotfiles/.bashrc

# Don't try to ping the internet, all the data is already on disk
# export HF_HUB_OFFLINE=1

# Activate environment
source $INF_PATH/virtualenvs/muvera_gnu/bin/activate

cd $DGX_PATH/CMUVERA_IR_ref

# Run your command
CUDA_VISIBLE_DEVICES=0 python3 -m src.colbert_embs k=10 method='augmented' num_rh_augment=8 data.dataset_name=$1 augment=True index=False dbl_norm=True embedder.mode='disk' colbert_topk=15 data.loader_type=$3 data.query_type=$4 &
CUDA_VISIBLE_DEVICES=1 python3 -m src.colbert_embs k=10 method='augmented' num_rh_augment=8 data.dataset_name=$1 augment=True index=False dbl_norm=True embedder.mode='disk' colbert_topk=20 data.loader_type=$3 data.query_type=$4 &
wait