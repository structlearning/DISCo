#!/bin/bash
#SBATCH --partition=dgx
#SBATCH --qos=dgx
#SBATCH --gpus=1             # Request 1 GPU
#SBATCH --cpus-per-task=64        # Request 64 CPUs
#SBATCH --mem=500G                # Request 500 GB RAM
#SBATCH --output=logs/muvera_$1_$2_%j.out  # Save stdout to logs/train-<jobid>.out
#SBATCH --error=logs/muvera_$1_$2_%j.err   # Save stderr to logs/train-<jobid>.err

# Load modules if necessary
# module load python/3.10
source ~/.bashrc

# Don't try to ping the internet, all the data is already on disk
export HF_HUB_OFFLINE=1

# Activate environment
source ~/muvera/bin/activate

cd ~/CMUVERA_IR_ref

# Run your command
python3 -m src.colbert_embs k=15 method='augmented' num_rh_augment=8 data.dataset_name=$1 augment=True index=False dbl_norm=True embedder.mode='disk' colbert_topk=$2