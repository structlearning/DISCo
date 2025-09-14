from loguru import logger
import numpy as np
import muvfde
import torch
import copy
from src.dataloader import get_dataloader
import torch
import os,pickle,socket,sys
import logging
from .utils import set_seed, load, save, hamming_distance
from torch import Tensor
from .embedder import ColBERTEmbedder
from tqdm import tqdm

from pebble import ProcessPool
from concurrent.futures import TimeoutError


def get_muvera(config, colbert_config):
    if config.muvera.type == "BERT":
        return MUVERA(config, colbert_config)
    else:
        raise ValueError("Invalid variety")


def _RH_augmentation_corpus(embs, colbert_config, config, embedder, ret_masks=True):
    # NOTE : INDRA
    if colbert_config.dbl_norm:
        embs[:,:,-1] = 0
        embs = torch.nn.functional.normalize(embs, p=2, dim=2)
    
    embs[:,:,-1] = -1
    augmented_embs = []
    new_masks = []

    for i in range(colbert_config.num_rh_augment):
        filename = f"./experiments/{config.data.dataset_name}/RH.{config.embedder.emb_dim}.{i}.pt"
        generate_new_rh = colbert_config.generate_new_rh
        # logger.info(f"Loading/Generating RH matrix from {filename}, generate_new_rh={generate_new_rh}")
        assert (generate_new_rh == False)
        assert filename is not None, "RH_file must be set in the config"
        # logger.info(f"Assertions done")
        if generate_new_rh:
            import hashlib
            # Hash the filename and use it as the seed for reproducibility
            seed = int.from_bytes(hashlib.sha256(filename.encode()).digest()[:4], 'big')
            gen = torch.Generator(device="cpu")
            gen.manual_seed(seed)
            RH = torch.randn(embs.size(2), generator=gen).to(embs.device)
            torch.save(RH, filename)
        else:
            assert os.path.exists(filename)
            RH = torch.load(filename).to('cpu')

        # print(f"RH device: {RH.device}, embs device: {embs.device}")
        signs = torch.sign(embs @ RH)
        signs[signs == 0] = 1
        reflect = signs.unsqueeze(-1)*embs
        augmented_embs.append(torch.cat([embs, reflect], dim=-1))

        if ret_masks:
            new_masks.append(embedder.cmasks)

    logger.info(f"Completed RH augmentation, returning {len(augmented_embs)} augmented copies")

    if ret_masks:
        return torch.cat(augmented_embs, dim = 0), torch.cat(new_masks, dim=0)
    else:
        return torch.cat(augmented_embs, dim = 0)


def _process_file(filename, self_config):
    """
    Worker function: process one embedding file and save MUVERA output.
    `self_config` is a plain dict with all paths + flags needed.
    """

    logger.info(f"Unpacking config for this worker {os.getpid()} for file {filename}")
    # Unpack config (avoid pickling `self`)
    embedding_parent_dir = self_config["embedding_parent_dir"]
    masks_parent_dir = self_config["masks_parent_dir"]
    muvera_path = self_config["muvera_path"]
    muvera_full_path = self_config["muvera_full_path"]
    muvera_aug_path = self_config["muvera_aug_path"]
    muvera_config = self_config["config"]
    colbert_config = self_config["colbert_config"]
    global_config = self_config["global_config"]
    embedder = self_config["embedder"]

    # Rebuild FDE generators inside the worker
    fde_generator_clean = FdeLateInteractionModel(
        muvera_config.num_repetitions,
        muvera_config.num_simhash_projections,
        muvera_config.projection_dimension,
        muvera_config.final_projection_dimension,
    )
    fde_generator_aug = FdeLateInteractionModel(
        muvera_config.num_repetitions,
        muvera_config.num_simhash_projections,
        muvera_config.projection_dimension,
        muvera_config.final_projection_dimension,
    )

    # Load embeddings + masks
    embs_dict = torch.load(os.path.join(embedding_parent_dir, filename), map_location="cpu", weights_only=False)
    masks_dict = torch.load(os.path.join(masks_parent_dir, filename), map_location="cpu", weights_only=False)
    logger.info(f"Worker {os.getpid()} loaded embeddings and masks for file {filename}")

    embs_dict_final = {k: v for k, v in embs_dict.items() if k != "embs_compressed"}
    cembs = embs_dict["embs_compressed"]
    cmasks = masks_dict["masks"]

    # Augmentation (if enabled)
    if global_config.augment:
        # logger.info(f"Worker {os.getpid()} performing augmentation for file {filename}")
        with torch.no_grad():
            augmented_cembs = _RH_augmentation_corpus(cembs, colbert_config, global_config, embedder, ret_masks=False)
            augmented_cembs = torch.nn.functional.normalize(augmented_cembs, p=2, dim=2)
            cembs = augmented_cembs.half()
            cmasks = cmasks.repeat(colbert_config.num_rh_augment, 1)
            assert cmasks.shape[0] == cembs.shape[0]

            # logger.info(f"After augmentation, cembs shape: {cembs.shape}, cmasks shape: {cmasks.shape}")
    else:
        if muvera_config.half_embs:
            cembs = cembs.half()

    # Encode each corpus item
    cembs_muvera = []
    for cidx, cemb in enumerate(tqdm(cembs, desc=f"Converting into Muvera encodings")):
        cmask = cmasks[cidx]
        cemb = cemb[cmask]  # remove padded tokens
        if not global_config.augment:
            cembs_muvera.append(fde_generator_clean.encode_single_item(cemb))
        else:
            cembs_muvera.append(fde_generator_aug.encode_single_item(cemb))

    cembs_muvera = np.stack(cembs_muvera, axis=0)

    # Save outputs
    if not global_config.augment:
        embs_dict_final["embs_muvera"] = cembs_muvera
        out_dir = muvera_path if muvera_config.half_embs else muvera_full_path
    else:
        embs_dict_final["embs_muvera_aug"] = cembs_muvera
        out_dir = muvera_aug_path

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    torch.save(embs_dict_final, out_path)

    logger.info(f"Worker {os.getpid()} saved Muvera embeddings to {out_path} for file {filename}")

    return out_path

    
# Wrapper to calculate MUVERA FDE
class FdeLateInteractionModel():
    def __init__(self, num_repetitions: int, num_simhash_projections: int, projection_dimension: int, final_projection_dimension: int | None = None, seed: int = 1221, **kwargs):
        super().__init__()  # empty init for Wrapper

        # query_config = muvfde.fixed_dimensional_encoding_config()
        # query_config.set_encoding_type(muvfde.encoding_type.DEFAULT_SUM)

        doc_config = muvfde.fixed_dimensional_encoding_config()
        doc_config.set_encoding_type(muvfde.encoding_type.AVERAGE)
        doc_config.enable_fill_empty(True)

        # for c in [query_config, doc_config]:
        for c in [doc_config]:
            c.set_num_repetitions(num_repetitions)
            c.set_num_simhash_projections(num_simhash_projections)
            c.set_projection_dimension(projection_dimension)
            c.set_seed(seed)
            if final_projection_dimension is not None:
                c.set_projection_type(muvfde.projection_type.DEFAULT_IDENTITY)
                c.set_final_projection_dimension(final_projection_dimension)
            else:
                c.set_projection_type(muvfde.projection_type.AMS_SKETCH)

        # self.q_cfg = query_config
        self.d_cfg = doc_config

    def encode(self, cembs) -> np.ndarray:
        # pick the right config
        cfg = self.d_cfg
        # compress each [seq_len × dim] matrix → [fde_dim]
        fde_out = []
        for mat in tqdm(cembs, desc="Encoding Corpus embeddings"):
            fde_out.append(
                muvfde.generate_fixed_dimensional_encoding(
                    mat.cpu().to(torch.float32), cfg
                )
            )
        return np.stack(fde_out, axis=0)

    def encode_single_item(self, cemb) -> np.ndarray:
        # pick the right config
        cfg = self.d_cfg
        return muvfde.generate_fixed_dimensional_encoding(
            cemb.cpu().to(torch.float32), cfg
        )

    def similarity(self, a: np.ndarray, b: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(a @ b.T)

    
class MUVERA:
    def __init__(self,config, colbert_config):
        
        self.global_config = config 
        self.colbert_config = colbert_config
        self.config = config.muvera
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataloader = get_dataloader(self.global_config.data)

        self.compress = self.config.compress
        if self.compress:
            self.compressed_size = self.config.compress_dim
            self.compression_mode = self.config.compression_mode

        self.embedder = ColBERTEmbedder(self.global_config.embedder)
        self.k = self.global_config.k
        self.dataset_name = self.global_config.data.dataset_name
        self.type = self.config.type

        self.prefix_str = f"./experiments/{self.dataset_name}/{self.type}"
        suffix_str = f"compressed_{self.global_config.lin_dim}"
        self.embedding_parent_dir = f"{self.prefix_str}/corpus/{suffix_str}"
        self.masks_parent_dir = f"{self.prefix_str}/corpus/masks"
        self.status_file  = f"{self.prefix_str}/corpus/status.json"

    def generate_fde(self):
        self.embedder.embed_full_dataset(self.dataloader,mode=self.global_config.embedder.mode) 
        logger.info("Full Dataset Embedded")
        self.batched = True
        self.query_num = len(self.embedder.qembs)
        set_seed(self.global_config.baseline.seed)
        
        print(self.embedder.cembs.shape)

        # Augmentation 
        with torch.no_grad():
            augmented_cembs, augmented_cmasks = self._RH_augmentation_corpus(self.embedder.cembs)
            augmented_cembs = torch.nn.functional.normalize(augmented_cembs, p=2, dim=2)
            augmented_cembs = augmented_cembs.half()
            print(augmented_cembs.shape)
            print(augmented_cmasks.shape)

        # fde_generator = FdeLateInteractionModel(augmented_cembs, augmented_cmasks, 20 ,5, 20)
        fde_generator = FdeLateInteractionModel(augmented_cembs, augmented_cmasks, self.config.num_repetitions,
                                                self.config.num_simhash_projections,
                                                self.config.projection_dimension,
                                                self.config.final_projection_dimension)
        fde = fde_generator.encode()

        print(fde.shape)

    def dump_fde_to_disk_parallel(self):
        """
        Parallelized version of dump_fde_to_disk.
        """
        # Build output paths
        muvera_path = f"{self.prefix_str}/corpus/compressed_muvera_{self.global_config.lin_dim}"
        muvera_full_path = f"{self.prefix_str}/corpus/compressed_muvera_full_{self.global_config.lin_dim}"
        muvera_aug_path = f"{self.prefix_str}/corpus/compressed_muvera_aug_{self.global_config.lin_dim}"

        os.makedirs(muvera_path, exist_ok=True)
        os.makedirs(muvera_full_path, exist_ok=True)
        os.makedirs(muvera_aug_path, exist_ok=True)

        # Gather filenames
        embed_filenames = [x for x in os.listdir(self.embedding_parent_dir) if x.endswith("pkl")]
        if self.global_config.augment:
            current_filenames = [x for x in os.listdir(muvera_aug_path) if x.endswith("pkl")]
        else:
            current_filenames = [x for x in os.listdir(muvera_path if self.config.half_embs else muvera_full_path)
                                if x.endswith("pkl")]
        if not self.config.fresh_start:
            embed_filenames = list(set(embed_filenames) - set(current_filenames))

        # Pack config into a plain dict (picklable)
        self_config = {
            "embedding_parent_dir": self.embedding_parent_dir,
            "masks_parent_dir": self.masks_parent_dir,
            "muvera_path": muvera_path,
            "muvera_full_path": muvera_full_path,
            "muvera_aug_path": muvera_aug_path,
            "config": self.config,
            "colbert_config": self.colbert_config,
            "global_config": self.global_config,
            "embedder": self.embedder
        }

        # Parallel execution
        results = []
        with ProcessPool(max_workers=10, max_tasks=1) as pool:
            future = pool.map(_process_file,
                              embed_filenames,
                              [self_config] * len(embed_filenames),
                              timeout=None, chunksize=1)

            it = future.result()
            for _ in range(len(embed_filenames)):
                try:
                    result = next(it)   # result is whatever _process_file returns
                    results.append(result)
                except TimeoutError:
                    # one worker hung, you can skip or retry
                    continue
                except Exception as e:
                    # one worker had an exception, you can skip or retry
                    logger.error(f"Worker had exception: {e}")
                    continue

        logger.info(f"All files processed. Processed {len(results)} files.")

    def dump_fde_to_disk(self):
        """
        Load ColBERT embeddings from disk, encode using Muvera and save FDE to disk.
        Saves two sets of embeddings to disk: Muvera applied on ColBERT embeddings and Muvera applied on ColBERT embeddings with RH augmentation.

        Note: To deal with padded tokens, we must manually filter them out before passing them onto MUVERA.
        This means we cannot pass a batch of corpus embeddings, and have to run a loop over each corpus item.
        """
        assert os.path.exists(self.embedding_parent_dir), f"Embedding directory {self.embedding_parent_dir} does not exist."
        muvera_path = f"{self.prefix_str}/corpus/compressed_muvera_{self.global_config.lin_dim}"
        muvera_full_path = f"{self.prefix_str}/corpus/compressed_muvera_full_{self.global_config.lin_dim}"
        muvera_aug_path = f"{self.prefix_str}/corpus/compressed_muvera_aug_{self.global_config.lin_dim}"

        os.makedirs(muvera_path, exist_ok=True)
        os.makedirs(muvera_full_path, exist_ok=True)
        os.makedirs(muvera_aug_path, exist_ok=True)

        embed_filenames = [x for x in os.listdir(self.embedding_parent_dir) if x.endswith("pkl")]

        if self.global_config.augment:
            current_filenames = [x for x in os.listdir(muvera_aug_path) if x.endswith("pkl")]
        else:
            if self.config.half_embs:
                current_filenames = [x for x in os.listdir(muvera_path) if x.endswith("pkl")]
            else:
                current_filenames = [x for x in os.listdir(muvera_full_path) if x.endswith("pkl")]

        if not self.config.fresh_start:
            remaining = list(set(embed_filenames) - set(current_filenames))
            embed_filenames = remaining

        set_seed(self.global_config.baseline.seed)

        fde_generator_clean = FdeLateInteractionModel(self.config.num_repetitions,
                                                      self.config.num_simhash_projections,
                                                      self.config.projection_dimension,
                                                      self.config.final_projection_dimension)
        fde_generator_aug = FdeLateInteractionModel(self.config.num_repetitions,
                                                    self.config.num_simhash_projections,
                                                    self.config.projection_dimension,
                                                    self.config.final_projection_dimension)

        for filename in tqdm(embed_filenames):
            embs_dict = torch.load(os.path.join(self.embedding_parent_dir, filename), weights_only=False)
            masks_dict = torch.load(os.path.join(self.masks_parent_dir, filename), weights_only=False)

            embs_dict_final = {}
            for key, value in embs_dict.items():
                if key != "embs_compressed":
                    embs_dict_final[key] = value

            cembs = embs_dict["embs_compressed"]
            cmasks = masks_dict["masks"]

            if self.global_config.augment:
                with torch.no_grad():
                    augmented_cembs = self._RH_augmentation_corpus(cembs, ret_masks=False)
                    augmented_cembs = torch.nn.functional.normalize(augmented_cembs, p=2, dim=2)
                    cembs = augmented_cembs.half()
                    # 8 * |C| x seq_len
                    cmasks = cmasks.repeat(self.colbert_config.num_rh_augment, 1)
                    assert cmasks.shape[0] == cembs.shape[0], \
                        f"cmasks shape {cmasks.shape} does not match cembs shape {cembs.shape} after augmentation"

            else:
                if self.config.half_embs:
                    cembs = cembs.half()

            cembs_muvera = []

            for cidx, cemb in enumerate(tqdm(cembs, desc=f"Processing {filename} Corpus Embeddings")):
                cmask = cmasks[cidx]
                cemb = cemb[cmask]  # Filter out padded tokens

                if not self.global_config.augment:
                    fde_clean = fde_generator_clean.encode_single_item(cemb)
                    cembs_muvera.append(fde_clean)

                else:
                    fde_aug = fde_generator_aug.encode_single_item(cemb)
                    cembs_muvera.append(fde_aug)

            cembs_muvera = np.stack(cembs_muvera, axis=0)
            if not self.global_config.augment:
                embs_dict_final["embs_muvera"] = cembs_muvera
                if self.config.half_embs:
                    torch.save(embs_dict_final, os.path.join(muvera_path, filename))
                    logger.info(f"Saved Muvera embeddings to {muvera_path} for file {filename}")
                else:
                    torch.save(embs_dict_final, os.path.join(muvera_full_path, filename))
                    logger.info(f"Saved Muvera embeddings to {muvera_full_path} for file {filename}")

            else:
                embs_dict_final["embs_muvera_aug"] = cembs_muvera
                torch.save(embs_dict_final, os.path.join(muvera_aug_path, filename))

                logger.info(f"Saved Muvera embeddings to {muvera_aug_path} for file {filename}")


if __name__=="__main__":
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    from omegaconf import OmegaConf
    os.makedirs("logs/muvera_fde_gen",exist_ok=True)
    
    file_config = OmegaConf.load("configs/config.yaml")
    colbert_config = OmegaConf.load("configs/colbert.yaml")
    cli_config = OmegaConf.from_cli()

    config = OmegaConf.merge(file_config,cli_config)
    logger.add(f"logs/muvera_fde_gen/muvera_{config.data.dataset_name}_{os.getpid()}.log", level="INFO")

    # logging.basicConfig(filename=f'logs/muvera_fde_gen/{config.method}_{config.data.dataset_name}.log', level=logging.INFO, format='%(asctime)s %(message)s')
    logger.info(f"COMMAND: {' '.join(sys.argv)}")
    logger.info(f"Running MUVERA FDE Generation with config: {OmegaConf.to_yaml(config)}")
    logger.info(f"This run is executing on {socket.gethostname()}")
    logger.info(f"PID: {os.getpid()}")
    # Run FDE Generation
    fde_gen = get_muvera(config, colbert_config)

    if config.embedder.mode == "mem":
        fde_gen.generate_fde()
    elif config.embedder.mode == "disk":
        if config.muvera.parallel:
            fde_gen.dump_fde_to_disk_parallel()
        else:
            fde_gen.dump_fde_to_disk()