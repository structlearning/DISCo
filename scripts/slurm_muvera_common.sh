#!/bin/bash

# Load modules if necessary
# module load python/3.10

# Don't try to ping the internet, all the data is already on disk
export HF_HUB_OFFLINE=1

# Activate environment
source ~/muvera/bin/activate

cd ~/CMUVERA_IR_ref

# Run your command
# python3 -m src.cmuvera data.dataset_name="scifact" overwrite_index=False index=False augment=True dbl_norm=True method="muvera" muvera.num_repetitions=20 muvera.num_simhash_projections=5 muvera.projection_dimension=20 muvera.final_projection_dimension=2560 lin_dim=128 muvera.half_embs=True muvera.type="BERT" muvera.compress=False embedder.mode="disk"
python3 -m src.cmuvera data.dataset_name="$1" overwrite_index=False index=False augment=$2 dbl_norm=True method="muvera" muvera.num_repetitions=$3 muvera.num_simhash_projections=$4 muvera.projection_dimension=$5 muvera.final_projection_dimension=$6 lin_dim=$7 muvera.half_embs=$8 muvera.fresh_start=$9 muvera.type="BERT" muvera.compress=False embedder.mode="disk" muvera.parallel=$10