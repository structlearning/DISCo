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
import submodlib

from .dataloader import get_dataloader
from .embedder import ColBERTEmbedder

from .utils import partial_chamfer_sim_batched_with_rerank


logger = logging.getLogger(__name__)
def get_method(config):
    name = config.method
    if name == "v0":
        return GreedyBaseline_v0(config)
    elif name=="sml":
        return GreedyBaseline_submodlib(config)
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
        self.variety = f"greedy_base_{config.baseline.bucket_size}_{self.config.embedder.emb_dim}_k{config.k}"
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


class GreedyBaseline_submodlib(BaseE2E):
    ## greedy routine with submodlib , exact
    def __init__(self,config):
        super().__init__(config)
        self.embedder = ColBERTEmbedder(config.embedder)
        # self.partial = None
        
        if config.submodlib.optimizer == "lazy":
            self.optimizer = "LazyGreedy"
        elif config.submodlib.optimizer == 'ltl':
            self.optimizer = "LazierThanLazyGreedy"
        elif config.submodlib.optimizer == 'naive':
            logger.warning("Naive optimizer is not recommended, proceeding with caution")
            self.optimizer = "NaiveGreedy"
        else : 
            raise ValueError(f"This optimizer is not allowed: {config.submodlib.optimizer}")
        self.variety = f"greedy_submodlib_{self.optimizer}_k{config.k}"
        self.k = config.k
    
    def get_single(self,qvec): # TODO can be multiprocessed?
        corp_size = len(self.embedder.cembs) ## embedder must be in mem mode
        corpus,masks = self.embedder.get_corpus(torch.arange(corp_size))
        logger.info(f"corpus shape : {corpus.shape}")
        logger.info(f"masks shape : {masks.shape}")
        logger.info(f"qvec shape : {qvec.shape}")
        
        qvec = qvec.to(corpus.device)
        # partial_chamfer = partial_chamfer_sim(qvec, corpus,masks)
        
        sb = (corpus)@(qvec.T)
        
        logger.info(f"sb {sb}; {sb.shape}")
        
        sb[masks==0] = -10
        # logger.info(f"sb masked {sb}")
        partial = sb.amax(dim=1).T
        # logger.info(f"partial {partial}")
        
        opt_for_each_query = submodlib.functions.facilityLocation.FacilityLocationFunction(n=corp_size,mode="dense",separate_rep=True,n_rep=len(qvec),sijs=np.array(partial.cpu().tolist()))
    
        # opt_for_each_query = submodlib.functions.facilityLocation.FacilityLocationFunction(n=self.config.baseline.bucket_size,mode="dense",separate_rep=True,n_rep=len(qvec),sijs=partial_chamfer)
        result = opt_for_each_query.maximize(budget=self.k,stopIfZeroGain=True,optimizer=self.optimizer)
        

        return result
        
        
    def run_old(self):
        result_path = f"./pickles/results/{self.variety}_{self.dataloader.dataset_name}_{self.config.retriever.type}{self.suffix}_k{self.k}.pkl"
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
        # chkpath = f"./pickles/results/{self.variety}_{self.dataloader.dataset_name}_{self.config.retriever.type}_chkpt.pkl"
        # chklogpath = f"./pickles/results/{self.variety}_{self.dataloader.dataset_name}_{self.config.retriever.type}_log.pkl"
        # if os.path.exists(chklogpath):
        #     chklog = load(chklogpath)
        #     opts = load(chkpath,"rb")
        #     chkpt = chklog["completed_qid"] + 1 
        #     set_seed_from_checkpoint(**chklog["seeds"])
            
        
        logger.info(f"Starting from query_id: {chkpt}")
        start = time.time()
        lap = start
        ## speed this up with multiprocessing bruh
        for query_id in tqdm(range(chkpt,self.query_num)):
            query = self.embedder.qembs[query_id][self.embedder.qmasks[query_id].bool()]
            result = self.get_single(query)
            opts.append([i[0] for i in result])
            
        end = time.time()
        logger.info(f"Total time taken for running submodlib: {end-start} seconds")
        logger.info(f"Now recomputing scores")
        start = time.time()
        max_len = max([len(i) for i in opts])
        opts_tensor = torch.tensor([opt_i + [opt_i[-1]]*(max_len-len(opt_i)) for opt_i in opts],dtype=torch.int64)
        logger.info(f"opts tensor: {opts_tensor}; max_len {max_len}")
        cembs_to_rescore, cmasks_to_rescore = self.embedder.get_corpus(opts_tensor)
        # q,k,corpus_set,emb; q,k,corpus_set;
        cembs_to_rescore, cmasks_to_rescore = cembs_to_rescore.to(self.device), cmasks_to_rescore.to(self.device)
        qemb_masked = self.embedder.qembs*(self.embedder.qmasks.unsqueeze(-1).to(self.embedder.qembs.device,self.embedder.qembs.dtype))
        qemb_masked = qemb_masked.to(self.device)
        sim_matrix = torch.where(
                        cmasks_to_rescore.bool().unsqueeze(-1),
                        torch.einsum("qkse,qce->qksc",cembs_to_rescore,qemb_masked),
                        -10   
                    )
        # logger.info(f"Sim matrix: {sim_matrix.shape}")
        partials = sim_matrix.amax(dim=2)
        # logger.info(f"Partial: {partials.shape}")
        # logger.info(f"Partial: {partials}")
        cum_partials = torch.cummax(partials,dim=1).values
        logger.info(f"Cum_partials: {cum_partials.shape}")
        logger.info(f"Cum_partials: {cum_partials}")
        scores = cum_partials.sum(dim=2)
        logger.info(f"Scores: {scores.shape}")
        logger.info(f"Scores: {scores}")
        opts = (opts_tensor.cpu(),scores.cpu())
        end = time.time()
        logger.info(f"Total time taken for rescoring: {end-start} seconds")
        save(opts,result_path)
        return opts

    def run(self):
        result_path = f"./pickles/results/{self.variety}_{self.dataloader.dataset_name}_{self.config.retriever.type}{self.suffix}_k{self.k}.pkl"
        if os.path.exists(result_path) and False:
            logger.info("Loading existing results")
            return load(result_path)
        
        self.embedder.embed_full_dataset(self.dataloader,mode=self.config.embedder.mode) 
        self.batched = True
        self.query_num = len(self.embedder.qembs)
        # fetch from checkpoints
        set_seed(self.config.baseline.seed)
        opts = []
        
        flattened_queries = self.embedder.qembs.reshape(-1,self.embedder.qembs.size(-1))[self.embedder.qmasks.reshape(-1).bool()].to(self.device)
        query_sizes = self.embedder.qmasks.sum(dim=-1)
        partials_list = []
        corp_size = 0
        for cemb,cmask in tqdm(self.embedder.iterate_over_batches(self.device,self.config.embedder.mode),desc="Corpus"):
            corp_size += cmask.size(0)
            prod = cemb@(flattened_queries.T)
            prod[~cmask.bool()] = -10
            partials_list.append(np.array(prod.amax(dim=1).cpu().tolist()))
            
        q_start = 0
        
        logger.info(f"Starting from query_id: 0")
        start = time.time()
        lap = start
        ## speed this up with multiprocessing bruh
        for query_id,size in tqdm(enumerate(query_sizes), desc="Query", total=self.query_num):
            partial = np.concatenate([elem[:,q_start:q_start+size]for elem in partials_list],axis=0)
            opt_for_each_query = submodlib.functions.facilityLocation.FacilityLocationFunction(n=corp_size,mode="dense",separate_rep=True,n_rep=size,sijs=partial.T)
    
            q_start += size
            # opt_for_each_query = submodlib.functions.facilityLocation.FacilityLocationFunction(n=self.config.baseline.bucket_size,mode="dense",separate_rep=True,n_rep=len(qvec),sijs=partial_chamfer)
            result = opt_for_each_query.maximize(budget=self.k,stopIfZeroGain=True,optimizer=self.optimizer)
            opts.append([i[0] for i in result])
            
        end = time.time()
        logger.info(f"Total time taken for running submodlib: {end-start} seconds")
        logger.info(f"Now recomputing scores")
        start = time.time()
        max_len = max([len(i) for i in opts])
        opts_tensor = torch.tensor([opt_i + [opt_i[-1]]*(max_len-len(opt_i)) for opt_i in opts],dtype=torch.int64)
        logger.info(f"opts tensor: {opts_tensor}; max_len {max_len}")
        
        cembs_to_rescore, cmasks_to_rescore = self.embedder.get_corpus(opts_tensor)
        # q,k,corpus_set,emb; q,k,corpus_set;
        cembs_to_rescore, cmasks_to_rescore = cembs_to_rescore.to(self.device), cmasks_to_rescore.to(self.device)
        qemb_masked = self.embedder.qembs*(self.embedder.qmasks.unsqueeze(-1).to(self.embedder.qembs.device,self.embedder.qembs.dtype))
        qemb_masked = qemb_masked.to(self.device)
        sim_matrix = torch.where(
                        cmasks_to_rescore.bool().unsqueeze(-1),
                        torch.einsum("qkse,qce->qksc",cembs_to_rescore,qemb_masked),
                        -10   
                    )
        # logger.info(f"Sim matrix: {sim_matrix.shape}")
        partials = sim_matrix.amax(dim=2)
        # logger.info(f"Partial: {partials.shape}")
        # logger.info(f"Partial: {partials}")
        cum_partials = torch.cummax(partials,dim=1).values
        logger.info(f"Cum_partials: {cum_partials.shape}")
        logger.info(f"Cum_partials: {cum_partials}")
        scores = cum_partials.sum(dim=2)
        logger.info(f"Scores: {scores.shape}")
        logger.info(f"Scores: {scores}")
        opts = (opts_tensor.cpu(),scores.cpu())
        end = time.time()
        logger.info(f"Total time taken for rescoring: {end-start} seconds")
        save(opts,result_path)
        return opts
  

if __name__=="__main__":
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)

    os.makedirs("logs/end_to_end",exist_ok=True)
    
    file_config = OmegaConf.load("configs/config.yaml")
    cli_config = OmegaConf.from_cli()
    
    config = OmegaConf.merge(file_config,cli_config)
    
    logging.basicConfig(filename=f'logs/end_to_end/{config.method}_{config.data.dataset_name}_{config.retriever.type}.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(process)d - %(message)s')
    # logger.log(config)
    
    retriever = get_method(config)
    retriever.run()
