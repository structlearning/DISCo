#!/bin/bash
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --gpus=1             # Request 1 GPU
#SBATCH --cpus-per-task=64        # Request 64 CPUs
#SBATCH --mem=500G                # Request 500 GB RAM
#SBATCH --output=logs/muvera-%j.out  # Save stdout to logs/train-<jobid>.out
#SBATCH --error=logs/muvera-%j.err   # Save stderr to logs/train-<jobid>.err

source ./scripts/slurm_muvera_common.sh $1 $2 $3 $4 $5 $6 $7 $8 $9