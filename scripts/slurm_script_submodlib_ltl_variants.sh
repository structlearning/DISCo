#!/bin/bash
#SBATCH --partition=l40
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --qos=l40
#SBATCH --gres=gpu:3             # Request 2 GPUs
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16        # Request 64 CPUs
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

echo "Arguments received: $1, $2, $3, $4, $5, $6, $7"

# Run your command
CUDA_VISIBLE_DEVICES=0 python3 -m src.endtoend k=15 method='sml' data.dataset_name=$1 embedder.mode="disk" submodlib.optimizer=ltl embedder.mv_type='colbertv2-plaid' load_state=$3 submodlib.path_suffix=$4 submodlib.stop_if_zero_gain=$5 submodlib.epsilon=0.1 data.loader_type=$7 data.query_type=$8 &
CUDA_VISIBLE_DEVICES=1 python3 -m src.endtoend k=15 method='sml' data.dataset_name=$1 embedder.mode="disk" submodlib.optimizer=ltl embedder.mv_type='colbertv2-plaid' load_state=$3 submodlib.path_suffix=$4 submodlib.stop_if_zero_gain=$5 submodlib.epsilon=0.5 data.loader_type=$7 data.query_type=$8 &
CUDA_VISIBLE_DEVICES=2 python3 -m src.endtoend k=15 method='sml' data.dataset_name=$1 embedder.mode="disk" submodlib.optimizer=ltl embedder.mv_type='colbertv2-plaid' load_state=$3 submodlib.path_suffix=$4 submodlib.stop_if_zero_gain=$5 submodlib.epsilon=0.9 data.loader_type=$7 data.query_type=$8 &
wait