#!/bin/bash
#SBATCH --partition=a40
#SBATCH --nodes=1
#SBATCH --nodelist=cn27-a40
#SBATCH --ntasks=1
#SBATCH --qos=a40
#SBATCH --gres=gpu:1             # Request 2 GPUs
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

echo "Arguments received: $1, $2, $3, $4, $5, $6, $7, $8"

# Run your command
CUDA_VISIBLE_DEVICES=0 python3 -m src.endtoend k=10 method='sml' data.dataset_name=$1 embedder.mode="disk" submodlib.optimizer=stoc embedder.mv_type='colbertv2-plaid' load_state=$3 submodlib.path_suffix=$4 submodlib.stop_if_zero_gain=$5 submodlib.epsilon=$6 data.loader_type=$7 data.query_type=$8 &
wait