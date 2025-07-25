import numpy as np
import muvfde
import torch
import copy
from src.dataloader import get_dataloader
import torch
import os,pickle
import logging
from .utils import set_seed, load, save, hamming_distance
from torch import Tensor
from .embedder import ColBERTEmbedder
from tqdm import tqdm

logger = logging.getLogger(__name__)

    
def get_muvera(config, colbert_config):
    if config.muvera.type == "base":
        return MUVERA(config, colbert_config)
    else:
        raise ValueError("Invalid variety")
    
# Wrapper to calculate MUVERA FDE
class FdeLateInteractionModel():
    def __init__(self, cembs, cmasks, num_repetitions: int, num_simhash_projections: int, projection_dimension: int, final_projection_dimension: int | None = None, seed: int = 1221, **kwargs):
        super().__init__()  # empty init for Wrapper

        self.cembs = cembs
        self.cmasks = cmasks

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

    def encode(self) -> np.ndarray:
        # pick the right config
        cfg = self.d_cfg
        # compress each [seq_len × dim] matrix → [fde_dim]
        fde_out = []
        for mat in tqdm(self.cembs, desc="Encoding Corpus embeddings"):
            fde_out.append(
                muvfde.generate_fixed_dimensional_encoding(
                    mat.cpu().to(torch.float32), cfg
                )
            )
        return np.stack(fde_out, axis=0)

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

        self.variety = f"base"
        self.embedder = ColBERTEmbedder(self.global_config.embedder)
        self.k = self.global_config.k
    
    def _RH_augmentation_corpus(self,embs):
        # NOTE : INDRA
        if self.colbert_config.dbl_norm:
            embs[:,:,-1] = 0
            embs = torch.nn.functional.normalize(embs, p=2, dim=2)
        
        embs[:,:,-1] = -1
        augmented_embs = []
        new_masks = []

        for i in range(self.colbert_config.num_rh_augment):
            filename = f"./experiments/{config.data.dataset_name}/RH.{config.embedder.emb_dim}.{i}.pt"
            generate_new_rh = self.colbert_config.generate_new_rh
            assert (generate_new_rh == False)
            assert filename is not None, "RH_file must be set in the config"
            if generate_new_rh:
                import hashlib
                # Hash the filename and use it as the seed for reproducibility
                seed = int.from_bytes(hashlib.sha256(filename.encode()).digest()[:4], 'big')
                gen = torch.Generator(device="cpu")
                gen.manual_seed(seed)
                self.RH = torch.randn(embs.size(2), generator=gen).to(embs.device)
                torch.save(self.RH, filename)
            else:
                assert os.path.exists(filename)
                self.RH = torch.load(filename).to("cpu")
                    
            signs = torch.sign(embs @ self.RH)
            signs[signs == 0] = 1
            reflect = signs.unsqueeze(-1)*embs
            augmented_embs.append(torch.cat([embs, reflect], dim=-1))
            new_masks.append(self.embedder.cmasks)
        
        return torch.cat(augmented_embs, dim = 0), torch.cat(new_masks, dim=0)

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

        fde_generator = FdeLateInteractionModel(augmented_cembs, augmented_cmasks, 20 ,5, 20)
        fde = fde_generator.encode()

        print(fde.shape)

if __name__=="__main__":
    from omegaconf import OmegaConf
    os.makedirs("logs/muvera_fde_gen",exist_ok=True)
    
    file_config = OmegaConf.load("configs/config.yaml")
    colbert_config = OmegaConf.load("configs/colbert.yaml")
    cli_config = OmegaConf.from_cli()
    
    config = OmegaConf.merge(file_config,cli_config)
    
    logging.basicConfig(filename=f'logs/muvera_fde_gen/{config.method}_{config.data.dataset_name}.log', level=logging.INFO, format='%(asctime)s %(message)s')

    # Run FDE Generation
    fde_gen = get_muvera(config, colbert_config)
    fde_gen.generate_fde()