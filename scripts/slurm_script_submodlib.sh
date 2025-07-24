#!/bin/bash
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --gpus=1             # Request 1 GPU
#SBATCH --cpus-per-task=64        # Request 64 CPUs
#SBATCH --mem=500G                # Request 500 GB RAM
#SBATCH --output=logs/muvera-%j.out  # Save stdout to logs/train-<jobid>.out
#SBATCH --error=logs/muvera-%j.err   # Save stderr to logs/train-<jobid>.err

# Load modules if necessary
# module load python/3.10

# Don't try to ping the internet, all the data is already on disk
export HF_HUB_OFFLINE=1

# Activate environment
source ~/muvera/bin/activate

cd ~/CMUVERA_IR_ref

echo "Arguments received: $1, $2, $3, $4"

# Run your command
python3 -m src.endtoend k=15 method='sml' data.dataset_name=$1 embedder.mode="disk" submodlib.optimizer=$2 embedder.mv_type='colbertv2-plaid' load_state=$3 submodlib.path_suffix=$4 submodlib.stop_if_zero_gain=$5