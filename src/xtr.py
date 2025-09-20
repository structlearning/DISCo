import torch
import logging
import warnings
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

from .dataloader import get_dataloader
from .embedder import ColBERTEmbedder
from .endtoend import BaseE2E
from .colbert_embs import DummyQueryForColbert
from .utils import partial_chamfer_sim, save

import sys
import os
from tqdm import tqdm

import warp
from warp.engine.utils.index_converter import convert_index


## top k retriever via WARP. retrieve topk via WARP then use our score function to get the final scores
class WarpBaseline(BaseE2E):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.dataloader = get_dataloader(self.config.data)
        self.embedder = ColBERTEmbedder(config.embedder)
        self.variety = self.config.embedder.mv_type
        
        self.mv_type = config.embedder.mv_type + "/norm" # Default COLBERT behaviour 
        with warp.infra.Run().context(warp.infra.RunConfig(nranks=1, experiment=f"xtr_{self.dataloader.dataset_name}")):
            colbert_config=warp.infra.ColBERTConfig(
                nbits=self.config.colbert.nbits,
                root="./xtr_beir_expts/",
                dim=self.config.embedder.emb_dim
            )

        if self.config.embedder.model == "google/xtr-base-en":
            self.suffix = "_xtr-base-en"
        else:
            self.suffix = ""

    def index(self):
        dataset_name = self.dataloader.dataset_name
        corpus_tsv_filename, _ = self.dataloader.get_tsv()
        nbits = self.config.colbert.nbits
        index_name = f"xtr_nbits={nbits}.noaug{self.suffix}"
        with warp.infra.Run().context(warp.infra.RunConfig(nranks=1, experiment=f"{self.config.data.dataset_name}/{self.config.embedder.type}/{self.mv_type}")):
            config = warp.infra.ColBERTConfig(
                nbits=nbits,
                root="./xtr_beir_expts/",
                dim=self.config.embedder.emb_dim,
                doc_maxlen=warp.engine.constants.DOC_MAXLEN,
                index_bsize=128,
                index_root=f"./experiments/{self.config.data.dataset_name}/BERT/{self.mv_type}/indexes"
            )
            # indexer = warp.Indexer(checkpoint="google/xtr-base-en", config=config)
            indexer = warp.Indexer(checkpoint=self.config.embedder.model, config=config)
            indexer.index(name=index_name, collection=corpus_tsv_filename, overwrite=self.config.overwrite_index)
        ## index needs to be converted into a warp index
        convert_index(os.path.join(config.index_root, index_name))

    def search(self, queries, k):
        # ranking = self.searcher._search_all_Q(queries, qembs, k).todict()
        # Q_cpu set to True if using WARP engine search
        ranking = self.searcher.search_all(queries, k, Q_cpu=True).todict()

        # Order of dict enumerate (keys,values) is same as order of queries, i.e., order of insertion
        # This is ensured python 3.7 onwards.
        res = []
        for _, values in ranking.items():
            ids, _, _ = zip(*values)
            res.append(ids) 
            
        max_len = max([len(val) for val in res])
        logger.info(f"Max length of retrieved ids comes out to: {max_len}. Average length of merged ids is {sum([len(val) for val in res])/len(res)}. Otherwise would have been {self.config.colbert_topk}")
        ## TODO:tensorize -- just initialise the tensor to -1 then do torch.where ==-1 tensor[:,:1]
        result_tensor = torch.zeros((len(queries), max_len), dtype=torch.int64)
        for i in range(len(queries)):
            current_set = list(res[i])
            current_set.extend([current_set[0]] * (max_len - len(current_set)))
            result_tensor[i] = torch.tensor(current_set)
        
        return result_tensor
    
    def run(self):
        # Init searcher
        dataset_name = self.dataloader.dataset_name
        nbits = self.config.colbert.nbits
        index_name = f"xtr_nbits={nbits}.noaug{self.suffix}"
        index_root = f"./experiments/{self.config.data.dataset_name}/BERT/{self.mv_type}/indexes"

        # with warp.infra.Run().context(warp.infra.RunConfig(nranks=1, experiment=f"{self.config.data.dataset_name}/{self.config.embedder.type}/{self.mv_type}")):
        #     config = warp.infra.ColBERTConfig(
        #         nbits=nbits,
        #         root="./xtr_beir_expts/",
        #         dim=self.config.embedder.emb_dim,
        #         doc_maxlen=warp.engine.constants.DOC_MAXLEN,
        #         index_bsize=128,
        #         index_root=index_root,
        #     )

        warp_config = warp.engine.config.WARPRunConfig(
            nranks=4,
            collection=self.config.data.loader_type, # beir or lotte
            dataset=dataset_name,
            type_="search", # irrelevant unless LoTTE
            datasplit="test",
            nbits=nbits,
            k=self.config.k, # is this the same as the search k?
            t_prime=None,
            bound=128,
            fused_ext=True,
            index_root_=index_root,
            index_name_=index_name,
        )

        corpus_tsv_filename, _ = self.dataloader.get_tsv()
        self.searcher = warp.Searcher(
            checkpoint=self.config.embedder.model,
            index=index_name,
            index_root=index_root,
            config=warp_config,
            warp_engine=True,
            collection=corpus_tsv_filename
        )

        result_file_path = f"./pickles/results/xtr_{self.variety}_{self.config.data.dataset_name}_k{self.config.k}{self.suffix}.pkl"

        self.embedder.embed_full_dataset(self.dataloader, mode=self.config.embedder.mode)
        # Below embeddings used for re-ranking
        qembs, qmasks = self.embedder.qembs, self.embedder.qmasks

        _, queries = self.dataloader.get_data()
        result_ids = self.search(queries, self.config.k)
        chamfer_scores = []

        if self.config.embedder.mode =="mem":
            # TODO: Compute Chamfer scores here.
            ## compute real scores with opts
            cembs, cmasks = self.embedder.get_corpus(result_ids)
        else:
            corpus = self.embedder.get_corpus(result_ids)
            logger.info("All required documents are loaded")
            cembs = []
            cmasks = []
            for q_id in tqdm(range(qembs.shape[0]), desc="Processing queries"):
                cemb, cmask = corpus[
                    (q_id * torch.ones(result_ids.shape[1], dtype=torch.long, device=corpus.device)),
                    torch.arange(result_ids.shape[1], dtype=torch.long, device=corpus.device)
                ]

                out = partial_chamfer_sim(qembs[q_id][qmasks[q_id].bool()], cemb, cmask, device=qembs.device, bs=1024)
                out_sum = out.sum(dim=0)
                sorted_inds = torch.argsort(out_sum, descending=True)
                sorted_out = out[:, sorted_inds]
                res = torch.cummax(sorted_out, dim=1)[0].sum(dim=0)
                chamfer_scores.append(res)

                cembs.append(cemb)
                cmasks.append(cmask)

            cembs = torch.stack(cembs)
            cmasks = torch.stack(cmasks)
            chamfer_scores = torch.stack(chamfer_scores)

        ## qembs: 300,64,128, qmasks: 300,64
        ## cembs: 300,100,330,128, cmasks: 300,100,330
        
        ## qembs is already masked out, and zeroed out
        # dp = torch.einsum("abc,adec->adbe",qembs,cembs.to(qembs.device))
        # masked = torch.where(cmasks.bool().to(qembs.device).unsqueeze(2),dp,-10)
        # partials = torch.cummax(torch.amax(masked,dim=3),dim=1)[0]
        # result_scores = torch.sum(partials,dim=2)

        result_scores = chamfer_scores
        
        save((result_ids, result_scores), result_file_path)
        return result_ids, result_scores


if __name__=="__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    config = OmegaConf.load("configs/colbert.yaml")
    cliconfig = OmegaConf.from_cli()
    
    conf = OmegaConf.merge(config, cliconfig)
    os.makedirs("logs/colbert", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(process)d - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/colbert/{conf.method}_{conf.data.dataset_name}_{conf.retriever.type}_pid:{os.getpid()}.log'),
            logging.StreamHandler()
        ]
    )
    logger.info(conf)

    assert conf.method.startswith("xtr")
    assert conf.augment == False
    logging.info(f"Going to run XTR iid")
    obj = WarpBaseline(conf)

    if conf.index:
        import time
        start = time.time()
        print("Starting Indexing")
        obj.index()
        end = time.time()
        print("Done in ", end-start)
    else:
        import time
        start = time.time()
        print("Starting Run")
        
        obj.run()
        end = time.time()
        
        print("Done in ", end-start)