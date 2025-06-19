import json
import os
import sys
import torch


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "fever"

    BASE_DIR = f"../experiments/{ds}/BERT/corpus/"
    EMBEDDINGS_DIR = BASE_DIR + "compressed_128/"
    embeding_path = lambda folder,batch,minibatch : f"../experiments/{ds}/BERT/corpus/{folder}/batch_{batch}.{minibatch}.pkl"

    with open(BASE_DIR+"status.json", 'r') as f:
        info_json = json.load(f)

    doc_id_to_batchinfo = {}
    num_batches = len(info_json) // 2
    id = 0
    for i in range(num_batches):
        j = 0
        while os.path.exists(embeding_path(f"compressed_128", i, j)):
            data = torch.load(embeding_path(f"compressed_128", i, j))
            num_docs = data['embs_compressed'].shape[0]
            doc_id_to_batchinfo = {
                **doc_id_to_batchinfo,
                **{k: (i, j, k - id) for k in range(id, id + num_docs)}
            }
            id+=num_docs
            j += 1
            print(i,j," done")

            del data
            torch.cuda.empty_cache()


    with open(f"../docid_to_batchinfo/docid_to_batchinfo_{ds}.json", 'w') as f:
        json.dump(doc_id_to_batchinfo, f)