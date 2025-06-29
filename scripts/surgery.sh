#!/bin/bash
if [ "$HOSTNAME" != "fox" ]; then
    echo "This script is intended to be run on the fox server."
    exit 1
fi

gpus=(2 3 4 0 1)
# threshold = 2 times number of gpus
threshold=$((2 * ${#gpus[@]}))
counter=0
for dataset in 'scifact' 'nfcorpus'; do
    for dblnorm in 'True' 'False'; do 
        for b in '10' '25' '50' '200'; do
            gpu=${gpus[$((counter % ${#gpus[@]}))]}
            CUDA_VISIBLE_DEVICES="$gpu" python3 -m src.colbert_embs augment=True k=15 colbert_internal.rerank_internal=False data.dataset_name="$dataset" dbl_norm="$dblnorm" & 
            counter=$((counter + 1))
            if [ $counter -eq $threshold ]; then
                wait
                counter=0
            fi
        done
    done
done
wait
exit
for dataset in 'scifact' 'nfcorpus'; do
    for dblnorm in 'True' 'False'; do 
        for b in '10' '25' '50' '100' '200'; do
            gpu=${gpus[$((counter % ${#gpus[@]}))]}
            CUDA_VISIBLE_DEVICES="$gpu" python3 -m src.colbert_embs augment=True k=15 data.dataset_name="$dataset" dbl_norm="$dblnorm" colbert_topk="$b" & 
            counter=$((counter + 1))
            if [ $counter -eq $threshold ]; then
                wait
                counter=0
            fi
        done
    done
done
for dataset in 'scifact' 'nfcorpus'; do
    for dblnorm in 'True' 'False'; do 
        # for b in '10' '25' '50' '100' '200'; do
        gpu=${gpus[$((counter % ${#gpus[@]}))]}
        CUDA_VISIBLE_DEVICES="$gpu" python3 -m src.colbert_embs augment=True k=15 colbert_internal.rerank_external=False data.dataset_name="$dataset" dbl_norm="$dblnorm" colbert_topk="$b" & 
        counter=$((counter + 1))
        if [ $counter -eq $threshold ]; then
            wait
            counter=0
        fi
        # done
    done
done 
wait