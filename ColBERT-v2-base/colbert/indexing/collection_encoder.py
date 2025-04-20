import torch
import os
from tqdm import tqdm
import json

from colbert.infra.run import Run
from colbert.utils.utils import print_message, batch


class CollectionEncoder():
    def __init__(self, config, checkpoint):
        self.config = config
        self.checkpoint = checkpoint

    def encode_passages(self, passages):
        Run().print(f"#> Encoding {len(passages)} passages..")

        if len(passages) == 0:
            return None, None

        with torch.inference_mode():
            embs, doclens = [], []

            # Batch here to avoid OOM from storing intermediate embeddings on GPU.
            # Storing on the GPU helps with speed of masking, etc.
            # But ideally this batching happens internally inside docFromText.
            for passages_batch in batch(passages, self.config.bsize * 50):
                embs_, doclens_ = self.checkpoint.docFromText(passages_batch, bsize=self.config.bsize,
                                                              keep_dims='flatten', showprogress=False)
                embs.append(embs_)
                doclens.extend(doclens_)

            embs = torch.cat(embs)

            # embs, doclens = self.checkpoint.docFromText(passages, bsize=self.config.bsize,
            #                                                   keep_dims='flatten', showprogress=(self.config.rank < 1))

        # with torch.inference_mode():
        #     embs = self.checkpoint.docFromText(passages, bsize=self.config.bsize,
        #                                        keep_dims=False, showprogress=(self.config.rank < 1))
        #     assert type(embs) is list
        #     assert len(embs) == len(passages)

        #     doclens = [d.size(0) for d in embs]
        #     embs = torch.cat(embs)

        return embs, doclens
    
    def encode_passages_modified(self, passages):
        Run().print(f"#> Encoding {len(passages)} passages..")

        if len(passages) == 0:
            return None, None

        with torch.inference_mode():
            embs, doclens = [], []

            # Batch here to avoid OOM from storing intermediate embeddings on GPU.
            # Storing on the GPU helps with speed of masking, etc.
            # But ideally this batching happens internally inside docFromText.
            for passages_batch in batch(passages, self.config.bsize * 50):
                embs_, doclens_ = self.checkpoint.docFromText_modified(
                    passages_batch,
                    bsize=self.config.bsize,
                    keep_dims="flatten",
                    showprogress=False,
                    pool_factor=self.config.pool_factor,
                    clustering_mode=self.config.clustering_mode,
                    protected_tokens=self.config.protected_tokens,
                )
                embs.append(embs_)
                doclens.extend(doclens_)

            embs = torch.cat(embs)

        return embs, doclens
    
    def encode_passages_from_disk(self, chunk_idx, file_prefix):
        status_file = os.path.join(file_prefix, "status.json")
        batch_count = 0
        if os.path.exists(status_file):
            with open(status_file, "r") as f:
                status_data = json.load(f)
            assert status_data.get(f"status.{chunk_idx}"), "Status file indicates encoding is not completed."
            Run().print(f"#> Status file indicates encoding of chunk {chunk_idx} is completed.")
            batch_count = status_data.get(f"num_batches.{chunk_idx}")
        else:
            raise FileNotFoundError(f"Status file {status_file} does not exist.")
        
        Run().print(f"#> Loading passages from {file_prefix}..")
        
        embs = []
        doclens = []

        compressed_dir = os.path.join(file_prefix, f"compressed_{self.config.lin_dim}")
        masks_dir = os.path.join(file_prefix, "masks")

        assert os.path.exists(compressed_dir), f"Directory {compressed_dir} does not exist."
        assert os.path.exists(masks_dir), f"Directory {masks_dir} does not exist."

        compressed_files = os.listdir(compressed_dir)
        mask_files = os.listdir(masks_dir)

        assert len(compressed_files) == len(mask_files), "Mismatch in number of compressed and mask files."
        
        with torch.inference_mode():
            for batch_num in tqdm(range(len(compressed_files)), total=len(compressed_files)):
                compressed_path = os.path.join(compressed_dir, f"batch_{chunk_idx}.{batch_num}.pkl")
                mask_path = os.path.join(masks_dir, f"batch_{chunk_idx}.{batch_num}.pkl")


                compressed_data = torch.load(compressed_path)
                mask_data = torch.load(mask_path)

                assert compressed_data["batch_num"] == mask_data["batch_num"], f"Batch number mismatch: {compressed_data['batch_num']} vs {mask_data['batch_num']}"
                assert compressed_data["batch_size"] == mask_data["batch_size"], f"Batch size mismatch: {compressed_data['batch_size']} vs {mask_data['batch_size']}"

                embs_, doclens_ = self.checkpoint.docFromBaseEmb(
                        compressed_data["embs_compressed"],
                        mask_data["masks"],
                        bsize=self.config.index_bsize,
                        # keep_dims="flatten",
                        showprogress=(not self.use_gpu),
                    )
                embs.append(embs_)
                doclens.extend(doclens_)
            embs = torch.cat(embs)
        return embs, doclens