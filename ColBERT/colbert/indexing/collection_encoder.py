import torch
import os
from tqdm import tqdm
import json
from colbert.infra.run import Run
from colbert.utils.utils import print_message, batch


class CollectionEncoder:
    def __init__(self, config, checkpoint):
        self.config = config
        self.checkpoint = checkpoint
        self.use_gpu = self.config.total_visible_gpus > 0

    def encode_passages(self, passages):
        Run().print(f"#> Encoding {len(passages)} passages..")

        if len(passages) == 0:
            return None, None

        with torch.inference_mode():
            embs, doclens = [], []

            # Batch here to avoid OOM from storing intermediate embeddings on GPU.
            # Storing on the GPU helps with speed of masking, etc.
            # But ideally this batching happens internally inside docFromText.
            for passages_batch in batch(passages, self.config.index_bsize * 50):
                embs_, doclens_ = self.checkpoint.docFromText(
                    passages_batch,
                    bsize=self.config.index_bsize,
                    keep_dims="flatten",
                    showprogress=(not self.use_gpu),
                    pool_factor=self.config.pool_factor,
                    clustering_mode=self.config.clustering_mode,
                    protected_tokens=self.config.protected_tokens,
                )
                embs.append(embs_)
                doclens.extend(doclens_)

            embs = torch.cat(embs)

            # embs, doclens = self.checkpoint.docFromText(passages, bsize=self.config.index_bsize,
            #                                                   keep_dims='flatten', showprogress=(self.config.rank < 1))

        # with torch.inference_mode():
        #     embs = self.checkpoint.docFromText(passages, bsize=self.config.index_bsize,
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
            for passages_batch in batch(passages, self.config.index_bsize * 50):
                embs_, doclens_ = self.checkpoint.docFromText_modified(
                    passages_batch,
                    bsize=self.config.index_bsize,
                    keep_dims="flatten",
                    showprogress=(not self.use_gpu),
                    pool_factor=self.config.pool_factor,
                    clustering_mode=self.config.clustering_mode,
                    protected_tokens=self.config.protected_tokens,
                )
                embs.append(embs_)
                doclens.extend(doclens_)

            embs = torch.cat(embs)
        return embs, doclens
    
    def encode_passages_to_dump(self,passages, chunk_idx, file_prefix):

        # Ensure the file_prefix directory and subdirectories exist
        assert os.path.exists(file_prefix), f"File {file_prefix} does not exist. Create it beforehand."       
        
        # os.makedirs(os.path.join(file_prefix, "full"), exist_ok=True)
        os.makedirs(os.path.join(file_prefix, f"compressed_{self.config.lin_dim}"), exist_ok=True)
        os.makedirs(os.path.join(file_prefix, "masks"), exist_ok=True)
        
        ## TODO: create some lock so that we do not have redundant computation in this function when there are concurrent runs
        status_file = os.path.join(file_prefix, "status.json")
        status_metadata = {}
        if os.path.exists(status_file):
            with open(status_file, "r") as f:
                status_metadata = json.load(f)
                Run().print(f"#> Status file found. Current status: {status_metadata}")
            if status_metadata.get(f"status.{chunk_idx}") == True:
                Run().print(f"#> Status file indicates encoding of chunk {chunk_idx} is already completed. Exiting.")
                return

        Run().print(f"#> Encoding and dumping {len(passages)} passages to {file_prefix}..")

        if len(passages) == 0:
            return

        with torch.inference_mode():
            batch_count = 0
            for batch_num, passages_batch in tqdm(
                enumerate(batch(passages, self.config.index_bsize * 50)), 
                disable=not self.use_gpu
            ):
                compressed, full, masks = self.checkpoint.docFromText_emb(
                    passages_batch,
                    bsize=self.config.index_bsize,
                    keep_dims=True,
                    showprogress=(not self.use_gpu),
                )

                metadata = {
                    "batch_num": batch_num, 
                    "chuck_idx": chunk_idx,
                    "batch_size": len(passages_batch)
                }

                torch.save(
                    {"embs_compressed": compressed, **metadata}, 
                    os.path.join(file_prefix, f"compressed_{self.config.lin_dim}", f"batch_{chunk_idx}.{batch_num}.pkl")
                )
                # torch.save(
                #     {"embs_full": full, **metadata}, 
                #     os.path.join(file_prefix, "full", f"batch_{chunk_idx}.{batch_num}.pkl")
                # )
                torch.save(
                    {"masks": masks, **metadata}, 
                    os.path.join(file_prefix, "masks", f"batch_{chunk_idx}.{batch_num}.pkl")
                )
                batch_count = batch_num+1
            # Update the status file to indicate completion
            Run().print(f"#> Encoding completed. Updating status file {status_file}..")
            with open(status_file, "w") as f:
                json.dump(status_metadata|{f"status.{chunk_idx}": True, f"num_batches.{chunk_idx}": batch_count}, f)
        return
    
    def encode_passages_from_disk(self,chunk_idx, file_prefix):

        # Check if the status file exists and verify its status
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
            for batch_num in tqdm(range(batch_count), total=batch_count):
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
                        pool_factor=self.config.pool_factor,
                        clustering_mode=self.config.clustering_mode,
                        protected_tokens=self.config.protected_tokens,
                    )
                embs.append(embs_)
                doclens.extend(doclens_)
            embs = torch.cat(embs)
        return embs, doclens
