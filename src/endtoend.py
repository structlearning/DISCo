import torch
import os
import numpy as np
from omegaconf import OmegaConf
import time
import logging
from .utils import set_seed,set_seed_from_checkpoint, load, save
from tqdm import tqdm
import random
from numpy.random import default_rng

from .dataloader import get_dataloader
from .embedder import ColBERTEmbedder

from .utils import partial_chamfer_sim_batched_with_rerank


logger = logging.getLogger(__name__)
def get_method(config):
    name = config.method
    if name == "v0":
        return GreedyBaseline_v0(config)
    else:
        raise ValueError(f"Unknown method: {name}")


## "Abstract" enough base class
class BaseE2E:
    def __init__(self,config):
        self.config = config
        self.dataloader = get_dataloader(config.data)
        
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batched = None
        try:
            self.checkpoints = config.checkpoints
        except:
            self.checkpoints = []
            
        self.suffix = "" if config.retriever.num_batches == -1 else f"upto_{config.retriever.num_batches}"
        
    def run(self):
        pass


## Our baseline that can do both exhaustive and stochastic greedy. set baseline.bucket_size to 0 for exhaustive
## TODO: The batched version has not been scrutinised
class GreedyBaseline_v0(BaseE2E):
    
    def __init__(self,config):
        super().__init__(config)
        self.embedder = ColBERTEmbedder(config.embedder)
        self.variety = f"greedy_base_{config.baseline.bucket_size}_{self.config.embedder.emb_dim}"
        self.k = config.k

    def get_single_exact(self,qvec):
        corp_size = len(self.embedder.cembs)
        corpus,masks = self.embedder.get_corpus(torch.arange(corp_size))
        
        qvec = qvec.to(corpus.device)
        
        logger.info(f"qvec : {qvec}")
        sb = (corpus)@(qvec.T)
        logger.info(f"sb {sb}")
        logger.info(f"masks {masks}")
        additive_masks = (masks.to(torch.float32)-1)*2
        sb = torch.where(masks.bool().unsqueeze(-1), sb , -10)
        logger.info(f"sb masked {sb}")
        partial = sb.amax(dim=1).T # q_set,num_corpus
        logger.info(f"partial {partial}")
        result_ours = []
        # k = 0
        sim2 = partial.sum(dim=0) # num_corpus
        logger.info(f"sim2 {sim2}")
        val,ind = torch.max(sim2,0)
        optvec = partial[:,ind]
        result_ours.append((ind.item(),val.item()))
        for i in range(1,self.k):
            sb = (corpus)@(qvec.T)
            additive_masks = (masks.to(torch.float32)-1)*2
            sb = sb + additive_masks.unsqueeze(-1)
            partial = sb.amax(dim=1).T # q_set,num_corpus
            partial = torch.maximum(partial,optvec.unsqueeze(1))
            sim2 = partial.sum(dim=0)
            val, ind = torch.max(sim2,0)
            optvec = torch.maximum(optvec,partial[:,ind])
            result_ours.append((ind.item(),val.item()))
        
        # compute using submodlib
        return result_ours
        
    
    def get_single(self,qvec): # TODO can be multiprocessed?
        corp_size = len(self.embedder.cembs) ## embedder must be in mem mode
        rng = default_rng()
        rindices = torch.tensor(rng.choice(corp_size,size=self.config.baseline.bucket_size,replace=False))
        corpus,masks = self.embedder.get_corpus(rindices)
        # partial_chamfer = partial_chamfer_sim_to_npy(qvec, *self.embedder.get_corpus(rindices))
        logger.debug(f"qvec shape : {qvec.shape}")
        logger.debug(f"norm of corpus {torch.norm(corpus,dim=-1)}")
        logger.debug(f"norm of query {torch.norm(qvec, dim=-1)}")
        # num_corpus,512,q_set
        qvec = qvec.to(corpus.device)
        sb = (corpus)@(qvec.T)
        
        logger.debug(f"sb {sb}")
        
        additive_masks = (masks.to(torch.float32)-1)*2
        sb = sb + additive_masks.unsqueeze(-1)
        logger.debug(f"sb masked {sb}")
        partial = sb.amax(dim=1).T # q_set,num_corpus
        logger.debug(f"partial {partial}")
        
        result_ours = []
        # k = 0
        sim2 = partial.sum(dim=0) # num_corpus
        logger.info(f"sim2 {sim2}")
        val,ind = torch.max(sim2,0)
        optvec = partial[:,ind]
        result_ours.append((rindices[ind].item(),val.item()))
        for i in range(1,self.k):
            rindices = torch.tensor(rng.choice(corp_size,size=self.config.baseline.bucket_size,replace=False))
            corpus,masks = self.embedder.get_corpus(rindices)
            sb = (corpus)@(qvec.T)
            additive_masks = (masks.to(torch.float32)-1)*2
            sb = sb + additive_masks.unsqueeze(-1)
            partial = sb.amax(dim=1).T # q_set,num_corpus
            partial = torch.maximum(partial,optvec.unsqueeze(1))
            sim2 = partial.sum(dim=0)
            val, ind = torch.max(sim2,0)
            optvec = torch.maximum(optvec,partial[:,ind])
            result_ours.append((rindices[ind].item(),val.item()))
        
        # compute using submodlib
        return result_ours
    
    def get_batch(self,qembs,qmasks):
        corp_size = len(self.embedder.cembs) ## embedder must be in mem mode
        rng = default_rng()
        
    
        optvec = -2*torch.ones(qembs.size(0),qembs.size(1),1).to(qembs.device)
        opt_indices = -torch.ones(qembs.size(0),self.k).to(qembs.device)
        opts_scores = -2000*torch.ones(qembs.size(0),self.k).to(qembs.device)
        running_scores = -2001*torch.ones((qembs.size(0),)).to(qembs.device)
        
        for i in tqdm(range(self.k), desc="K", total=self.k):
            rindices = torch.tensor([rng.choice(corp_size,size=self.config.baseline.bucket_size,replace=False) for _ in range(qembs.size(0))])
            cemb,cmask = self.embedder.get_corpus(rindices)
        
            greedyvec , max_sim_indices, max_sim_scores = partial_chamfer_sim_batched_with_rerank(query=qembs,query_masks=qmasks,max_gain_corpus=cemb,max_gain_corpus_masks=cmask, running_optvec=optvec)
            actual_doc_ids = rindices[torch.arange(len(max_sim_indices)), max_sim_indices]
            
            optvec = torch.maximum(optvec,greedyvec.unsqueeze(-1).to(optvec.device))

            opt_indices[:,i].copy_(actual_doc_ids,non_blocking=True)
            opts_scores[:, i].copy_(max_sim_scores, non_blocking=True)

    def get_batch_exact(self,qembs,qmasks):
        corp_size = len(self.embedder.cembs) ## embedder must be in mem mode
    
        optvec = -2*torch.ones(qembs.size(0),qembs.size(1),1).to(qembs.device)
        opt_indices = -torch.ones(qembs.size(0),self.k).to(qembs.device)
        opts_scores = -2000*torch.ones(qembs.size(0),self.k).to(qembs.device)
        running_scores = -2001*torch.ones((qembs.size(0),)).to(qembs.device)
        
        for i in tqdm(range(self.k), desc="K", total=self.k):
            cemb,cmask = self.embedder.get_corpus(torch.arange(corp_size))
        
            greedyvec , max_sim_indices, max_sim_scores = partial_chamfer_sim_batched_with_rerank(query=qembs,query_masks=qmasks,max_gain_corpus=cemb,max_gain_corpus_masks=cmask, running_optvec=optvec)
            
            optvec = torch.maximum(optvec,greedyvec.unsqueeze(-1).to(optvec.device))

            opt_indices[:,i].copy_(max_sim_indices,non_blocking=True)
            opts_scores[:, i].copy_(max_sim_scores, non_blocking=True)
            
    #TODO: NOT TESTED OUT
    def run_batched(self):
        result_path = f"./pickles/results/{self.variety}_{self.dataloader.dataset_name}_{self.config.retriever.type}{self.suffix}.pkl"
        if os.path.exists(result_path) and False:
            logger.info("Loading existing results")
            return load(result_path)
        self.embedder.embed_full_dataset(self.dataloader)
        self.batched = True
        self.query_num = len(self.embedder.qembs)
        # fetch from checkpoints
        chkpt = 0
        set_seed(self.config.baseline.seed)
        opts = []
        chkpath = f"./pickles/results/{self.variety}_{self.dataloader.dataset_name}_{self.config.retriever.type}_chkpt.pkl"
        chklogpath = f"./pickles/results/{self.variety}_{self.dataloader.dataset_name}_{self.config.retriever.type}_log.pkl"
        if os.path.exists(chklogpath) and False:
            chklog = load(chklogpath)
            opts = load(chkpath,"rb")
            chkpt = chklog["completed_qid"] + 1 
            set_seed_from_checkpoint(**chklog["seeds"])
            
        
        logger.info(f"Starting from query_id: {chkpt}")
        start = time.time()
        lap = start
        
        if self.config.baseline.bucket_size==0: # exact
            for i,(q,m) in enumerate(self.embedder.get_query_batched(self.config.baseline.batch_size,self.device,start=chkpt)):
                opts.append(self.get_batch_exact(q,m))
                if (time.time() - lap)> 60*60 : # checkpoint every hour
                    chkpt += q.size(0)
                    torch.cuda.synchronize()
                    opts = [torch.cat(opts,dim=0)]
                    save(opts[0],chkpath)
                    save({
                            "completed_qid":chkpt-1, 
                            "seeds": {
                            "random_seed": random.getstate(),
                            "np_random__seed": np.random.get_state(),
                            "torch_random_seed": torch.get_rng_state()}
                        },chklogpath)
                    logger.info(f"Checkpoint at query_id: {chkpt-1} out of {self.query_num}")
                    lap = time.time()
        else: # random
            for i,(q,m) in enumerate(self.embedder.get_query_batched(self.config.baseline.batch_size,self.device,start=chkpt)):
                opts.append(self.get_batch(q,m))
                if (time.time() - lap)> 60*60 : # checkpoint every hour
                    chkpt += q.size(0)
                    torch.cuda.synchronize()
                    opts = [torch.cat(opts,dim=0)]
                    save(opts[0],chkpath)
                    save({
                            "completed_qid":chkpt-1, 
                            "seeds": {
                            "random_seed": random.getstate(),
                            "np_random__seed": np.random.get_state(),
                            "torch_random_seed": torch.get_rng_state()}
                        },chklogpath)
                    logger.info(f"Checkpoint at query_id: {chkpt-1} out of {self.query_num}")
                    lap = time.time()
        end = time.time()
        logger.info(f"Total time taken for running : {end-start} seconds")
        opts = torch.cat(opts,dim=0)
        save(opts,result_path)
        return opts
        
        
    def run(self):
        result_path = f"./pickles/results/{self.variety}_{self.dataloader.dataset_name}_{self.config.retriever.type}{self.suffix}.pkl"
        if os.path.exists(result_path) and False:
            logger.info("Loading existing results")
            return load(result_path)
        
        self.embedder.embed_full_dataset(self.dataloader,mode=self.config.embedder.mode) 
        self.batched = True
        self.query_num = len(self.embedder.qembs)
        # fetch from checkpoints
        chkpt = 0
        set_seed(self.config.baseline.seed)
        opts = []
        chkpath = f"./pickles/results/{self.variety}_{self.dataloader.dataset_name}_{self.config.retriever.type}_chkpt.pkl"
        chklogpath = f"./pickles/results/{self.variety}_{self.dataloader.dataset_name}_{self.config.retriever.type}_log.pkl"
        if os.path.exists(chklogpath):
            chklog = load(chklogpath)
            opts = load(chkpath,"rb")
            chkpt = chklog["completed_qid"] + 1 
            set_seed_from_checkpoint(**chklog["seeds"])
            
        
        logger.info(f"Starting from query_id: {chkpt}")
        start = time.time()
        lap = start
        ## speed this up with multiprocessing bruh
        if self.config.baseline.bucket_size==0: # exact
            for query_id in tqdm(range(chkpt,self.query_num)):
                query = self.embedder.qembs[query_id][self.embedder.qmasks[query_id].bool()]
                opts.append(self.get_single_exact(query))
                if (time.time() - lap)> 60*60 : # checkpoint every hour
                    save(opts,chkpath)
                    save({
                            "completed_qid":query_id, 
                            "seeds": {
                            "random_seed": random.getstate(),
                            "np_random__seed": np.random.get_state(),
                            "torch_random_seed": torch.get_rng_state()}
                        },chklogpath)
                    logger.info(f"Checkpoint at query_id: {query_id} out of {self.query_num}")
                    lap = time.time()
        else: # random
            for query_id in tqdm(range(chkpt,self.query_num)):
                query = self.embedder.qembs[query_id][self.embedder.qmasks[query_id].bool()]
                opts.append(self.get_single(query))
                if (time.time() - lap)> 60*60 : # checkpoint every hour
                    save(opts,chkpath)
                    save({
                            "completed_qid":query_id, 
                            "seeds": {
                            "random_seed": random.getstate(),
                            "np_random__seed": np.random.get_state(),
                            "torch_random_seed": torch.get_rng_state()}
                        },chklogpath)
                    logger.info(f"Checkpoint at query_id: {query_id} out of {self.query_num}")
                    lap = time.time()
        end = time.time()
        logger.info(f"Total time taken for running : {end-start} seconds")
        save(opts,result_path)
        return opts


if __name__=="__main__":

    os.makedirs("logs/end_to_end",exist_ok=True)
    
    file_config = OmegaConf.load("configs/config.yaml")
    cli_config = OmegaConf.from_cli()
    
    config = OmegaConf.merge(file_config,cli_config)
    
    logging.basicConfig(filename=f'logs/end_to_end/{config.method}_{config.data.dataset_name}_{config.retriever.type}.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(process)d - %(message)s')
    # logger.log(config)
    
    retriever = get_method(config)
    retriever.run()
