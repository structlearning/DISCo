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
import pickle

from .dataloader import get_dataloader
from .embedder import ColBERTEmbedder

from .utils import partial_chamfer_sim_batched_with_rerank

# 1) Make sure the env var is set *inside* Python too
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

# 2) Turn on PyTorch’s deterministic‐only mode
torch.use_deterministic_algorithms(True)


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
        self.global_idx = 0

    def get_single_exact(self,qvec):
        corp_size = len(self.embedder.cembs)
        corpus,masks = self.embedder.get_corpus(torch.arange(corp_size))
        
        qvec = qvec.to(corpus.device)
        
        logger.info(f"qvec : {qvec}")
        sb = (corpus)@(qvec.T)
        logger.info(f"sb {sb}")
        logger.info(f"masks {masks}")
        additive_masks = (masks.to(torch.float32)-1)*2
        # TODO: Maybe compute sb corpus-query product only once
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

    def get_single_exact_numpy(self, qvec):
        """
        This function is similar to get_single_exact but uses numpy for computations.
        It is not used in the current implementation but can be useful for debugging or comparison.
        """
        corp_size = len(self.embedder.cembs)
        corpus,masks = self.embedder.get_corpus(torch.arange(corp_size))

        corpus_np = corpus.cpu().numpy()
        masks_np = masks.cpu().numpy()

        logger.info(f"qvec : {qvec}")
        sb = (corpus_np)@(qvec.T)
        logger.info(f"sb {sb}")
        logger.info(f"masks {masks}")

        # Ensure masks_np is boolean and broadcast to sb's shape
        mask_expanded = masks_np.astype(bool)[..., np.newaxis]  # shape [B, T, 1]
        # Apply masking
        sb = np.where(mask_expanded, sb, -10)
        logger.info(f"sb masked {sb}")

        partial = np.amax(sb, axis=1).T # q_set,num_corpus
        logger.info(f"partial {partial}")

        result_ours = []
        # k = 0
        sim2 = partial.sum(axis=0) # num_corpus
        logger.info(f"sim2 {sim2}")
        val, ind = np.max(sim2, axis=0), np.argmax(sim2, axis=0)

        optvec = partial[:,ind]
        result_ours.append((ind,val))

        for i in range(1,self.k):
            sb = (corpus_np)@(qvec.T)
            additive_masks = ((masks.to(torch.float32)-1)*2).numpy()
            sb = sb + np.expand_dims(additive_masks, axis=-1)
            partial = np.amax(sb, axis=1).T # q_set,num_corpus
            partial = np.maximum(partial, np.expand_dims(optvec, axis=1))
            sim2 = partial.sum(axis=0)
            val, ind = np.max(sim2, axis=0), np.argmax(sim2, axis=0)
            optvec = np.maximum(optvec, partial[:,ind])
            result_ours.append((ind,val))
        
        # compute using submodlib
        return result_ours
    
    def get_single(self,qvec): # TODO can be multiprocessed?
        corp_size = len(self.embedder.cembs) ## embedder must be in mem mode
        rng = default_rng(43)
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
        
        rng = default_rng(43)
        
        optvec = -2*torch.ones(qembs.size(0),qembs.size(1)).to(qembs.device)
        opt_indices = -torch.ones(qembs.size(0),self.k).to(qembs.device)
        opts_scores = -2000*torch.ones(qembs.size(0),self.k).to(qembs.device)
        
        for i in tqdm(range(self.k), desc="K", total=self.k):
            if self.config.embedder.mode=="mem":
                corp_size = len(self.embedder.cembs) ## embedder must be in mem mode
                rindices = torch.tensor([rng.choice(corp_size,size=self.config.baseline.bucket_size,replace=False) for _ in range(qembs.size(0))])
                save(rindices, "./rindices_mem.pkl")
                cembs,cmasks = self.embedder.get_corpus(rindices)
            
                max_sim_partial, max_sim_indices, max_sim_scores = partial_chamfer_sim_batched_with_rerank(
                        qembs, qmasks, optvec.unsqueeze(-1), cembs, cmasks
                    )
                real_indices = rindices[torch.arange(rindices.size(0)), max_sim_indices]
                optvec = torch.maximum(optvec, max_sim_partial.to(optvec.device))
                opts_scores[:,i].copy_(max_sim_scores)
                opt_indices[:,i].copy_(real_indices)
            else:
                corp_size = len(self.embedder.docid_to_batchinfo.keys()) ## embedder must be in mem mode
                rindices = torch.tensor([rng.choice(corp_size,size=self.config.baseline.bucket_size,replace=False) for _ in range(qembs.size(0))])
                # save(rindices, "./rindices_disk.pkl")
                corpus = self.embedder.get_corpus(rindices)
                logger.info("All required documents are loaded")
                for q_id in tqdm(range(qembs.shape[0]), desc="Processing queries"):
                    cemb, cmask = corpus[
                        (q_id * torch.ones(rindices.shape[1], dtype=torch.long, device=corpus.device)),
                        torch.arange(rindices.shape[1], dtype=torch.long, device=corpus.device)
                    ]

                    max_sim_partial, max_sim_indices, max_sim_scores = partial_chamfer_sim_batched_with_rerank(
                        qembs[q_id].unsqueeze(0), qmasks[q_id].unsqueeze(0), optvec.unsqueeze(-1)[q_id].unsqueeze(0), cemb.unsqueeze(0), cmask.unsqueeze(0)
                    )
                    real_indices = rindices[q_id, max_sim_indices.cpu()]
                    optvec[q_id] = torch.maximum(optvec[q_id], max_sim_partial.squeeze(0).to(optvec.device))
                    opts_scores[q_id,i] = max_sim_scores
                    opt_indices[q_id,i] = real_indices

                    del cemb, cmask

        
        opt_indices = opt_indices.cpu()
        opts_scores = opts_scores.cpu()
        # Combine into list of tuples
        opts = [
            list(zip(opt_indices[i].tolist(), opts_scores[i].tolist()))
            for i in range(opt_indices.shape[0])
        ]

        return opts

    def get_batch_exact(self,qembs,qmasks):
        raise ValueError("Function Not Verified. Do not used")
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
        raise ValueError("Function Not Verified. Do not used")
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
            if self.config.embedder.mode=="mem":
                for query_id in tqdm(range(chkpt,self.query_num)):
                    query = self.embedder.qembs[query_id][self.embedder.qmasks[query_id].bool()]
                    opts.append(self.get_single_exact(query))
                    # opts.append(self.get_single_exact_numpy(query.cpu().numpy()))
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
            else: # if embedder.mode = disk
                if not self.config.baseline.distributed_search:
                    optvec = -2.0*torch.ones_like(self.embedder.qmasks).to(self.embedder.qembs.device)
                    opt_indices = -torch.ones(self.embedder.qembs.size(0),self.k).to(self.embedder.qembs.device)
                    opts_scores = -2000.0*torch.ones(self.embedder.qembs.size(0),self.k).to(self.embedder.qembs.device)

                    for i in tqdm(range(self.k), desc="K", total=self.k):
                        temp_optvec = -2.0*torch.ones_like(self.embedder.qmasks).to(self.embedder.qembs.device)
                        temp_opt_indices = -torch.ones(self.embedder.qembs.size(0)).to(self.embedder.qembs.device)
                        temp_opts_scores = -2000.0*torch.ones(self.embedder.qembs.size(0)).to(self.embedder.qembs.device)
                        doc_id = 0
                        # Iterate over (batch, mini-batch, chunk)
                        for cemb, cmask in tqdm(self.embedder.iterate_over_batches(self.device,self.config.embedder.mode),desc="Corpus"):      
                            # A single chunk's worth of indices
                            inds = torch.arange(doc_id, doc_id+cemb.shape[0])
                            for q_id in tqdm(range(self.embedder.qembs.shape[0]), desc="Processing queries"):
                                max_sim_partial, max_sim_indices, max_sim_scores = partial_chamfer_sim_batched_with_rerank(
                                    self.embedder.qembs[q_id].unsqueeze(0), self.embedder.qmasks[q_id].unsqueeze(0), optvec.unsqueeze(-1)[q_id].unsqueeze(0), cemb.unsqueeze(0), cmask.unsqueeze(0)
                                )
                                max_sim_indices = max_sim_indices.to("cpu")
                                real_indices = inds[max_sim_indices]
                                if torch.sum(max_sim_partial) > torch.sum(temp_optvec[q_id]):
                                    temp_optvec[q_id] = max_sim_partial.squeeze(0).to(optvec.device)

                                max_sim_scores = max_sim_scores.to(self.embedder.qembs.device)
                                if max_sim_scores > temp_opts_scores[q_id]:
                                    temp_opts_scores[q_id] = max_sim_scores
                                    temp_opt_indices[q_id] = real_indices

                            doc_id+=cemb.shape[0]
                            
                        opt_indices[:,i].copy_(temp_opt_indices,non_blocking=True)
                        opts_scores[:, i].copy_(temp_opts_scores, non_blocking=True)
                        optvec = torch.maximum(optvec, temp_optvec)
                        # print(optvec)
                        # with open(f"./disk_optvec_{i}.pkl", "wb") as f:
                        #     pickle.dump(optvec, f)
                    
                    opt_indices = opt_indices.cpu()
                    opts_scores = opts_scores.cpu()

                    # print(opt_indices)

                    # Combine into list of tuples - the way code is written earlier
                    opts = [
                        list(zip(opt_indices[i].tolist(), opts_scores[i].tolist()))
                        for i in range(opt_indices.shape[0])
                    ]
                else:
                    # opts = distributed_search(self.embedder, self.k, self.config.embedder.mode)
                    pass

        else: # random
            if self.config.embedder.mode=="mem":
                # for query_id in tqdm(range(chkpt,self.query_num)):
                #     query = self.embedder.qembs[query_id][self.embedder.qmasks[query_id].bool()]
                #     opts.append(self.get_single(query))
                #     if (time.time() - lap)> 60*60 : # checkpoint every hour
                #         save(opts,chkpath)
                #         save({
                #                 "completed_qid":query_id, 
                #                 "seeds": {
                #                 "random_seed": random.getstate(),
                #                 "np_random__seed": np.random.get_state(),
                #                 "torch_random_seed": torch.get_rng_state()}
                #             },chklogpath)
                #         logger.info(f"Checkpoint at query_id: {query_id} out of {self.query_num}")
                #         lap = time.time()
                qembs, qmasks = self.embedder.qembs, self.embedder.qmasks
                opts = self.get_batch(qembs,qmasks)
            else: # mode = disk        
                qembs, qmasks = self.embedder.qembs, self.embedder.qmasks
                opts = self.get_batch(qembs,qmasks)
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
        raise ValueError("Function not used")
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
        raise ValueError("Old version - DO NOT USE!!")
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

        if self.config.embedder.mode=="mem":
            for cemb,cmask in tqdm(self.embedder.iterate_over_batches(self.device,self.config.embedder.mode),desc="Corpus"):
                corp_size += cmask.size(0)
                prod = cemb@(flattened_queries.T)
                prod[~cmask.bool()] = -10
                partials_list.append(np.array(prod.amax(dim=1).cpu().tolist()))
                
        q_start = 0

        if self.config.embedder.mode=="mem":
            logger.info(f"Starting from query_id: 0")
            start = time.time()
            lap = start
            ## speed this up with multiprocessing bruh
            for query_id,size in tqdm(enumerate(query_sizes), desc="Query", total=self.query_num):
                partial = np.concatenate([elem[:,q_start:q_start+size]for elem in partials_list],axis=0)
                # submodlib issue: submodlib (according to the docs) expects the kernel matrix input to be of shape
                # (n_rep, n) where n_rep is size of representative set (query tokens) and n is the number of corpus documents.
                # However, the facility location implementation in submodlib expects the input to be in column-major order,
                # and reads in this order. This does not match the order of the input we provide, which is in row-major order.
                # In our fork of submodlib, the facility location function object reads the kernel matrix input in row-major order.
                # Note: We tested different combinations of partial vs partial.T, n vs n_rep setting, and row vs column-major input reading.
                # The only combination that gives correct output (that matches greedy baseline) is partial.T with n_rep = size and
                # n = corp_size, and row-major input reading.
                opt_for_each_query = submodlib.functions.facilityLocation.FacilityLocationFunction(n=corp_size,mode="dense",separate_rep=True,n_rep=size,sijs=partial.T)
        
                q_start += size
                # opt_for_each_query = submodlib.functions.facilityLocation.FacilityLocationFunction(n=self.config.baseline.bucket_size,mode="dense",separate_rep=True,n_rep=len(qvec),sijs=partial_chamfer)
                result = opt_for_each_query.maximize(budget=self.k,stopIfZeroGain=True,optimizer=self.optimizer)
                opts.append([i[0] for i in result])
        else: # mode = disk
            query_sizes = self.embedder.qmasks.sum(dim=-1)
            ## MULTIPLE DISK PROBE VARIANT
            mega_q_batch_size = 100
            start = time.time()
            for i in range(0,self.query_num,mega_q_batch_size):
                total_start = time.time()
                flattened_queries = self.embedder.qembs[i:i+mega_q_batch_size].reshape(-1,self.embedder.qembs.size(-1))[self.embedder.qmasks[i:i+mega_q_batch_size].reshape(-1).bool()].to(self.device)
                
                partials_list = []
                corp_size = 0
                batch_size = 10000
                for cemb,cmask in tqdm(self.embedder.iterate_over_batches(self.device,self.config.embedder.mode),desc="Corpus"):
                    corp_size += cmask.size(0)
                    
                    partial = torch.zeros((cemb.size(0),flattened_queries.size(0)),device='cpu')
                    for j in range(0,flattened_queries.size(0),batch_size):
                        partial[:,j:j+batch_size] = torch.where(cmask.bool().unsqueeze(-1),cemb@flattened_queries[j:j+batch_size].T,-10).amax(dim=1).cpu()
                    partials_list.append(partial.numpy())
                logger.info(F"Time for disk probe: {time.time()-total_start}")
                logger.info(f"Starting from query_id: {i}")
                lap = time.time()
                
                q_start = 0
                for query_id in tqdm(range(i,i+mega_q_batch_size), desc=f"Query batch {i}", total=mega_q_batch_size):
                    q_start_time = time.time()
                    size = query_sizes[query_id].item()
                    partial = np.concatenate([elem[:,q_start:q_start+size]for elem in partials_list],axis=0)
                    opt_for_each_query = submodlib.functions.facilityLocation.FacilityLocationFunction(n=corp_size,mode="dense",separate_rep=True,n_rep=size,sijs=partial.T)
            
                    q_start += size                    
                    result = opt_for_each_query.maximize(budget=self.k,stopIfZeroGain=True,optimizer=self.optimizer)
                    
                    opts.append([i[0] for i in result])
                    q_end_time = time.time()
                    logger.info(f"Total time taken for query {query_id}: {q_end_time-q_start_time} seconds")
                end = time.time()
                logger.info(f"Total time taken for running submodlib/querying: {end-lap} seconds")
            end = time.time()
            logger.info(f"Total time taken for running both steps: {end-start} seconds")


        logger.info(f"Now recomputing scores")
            
        end = time.time()
        logger.info(f"Total time taken for running submodlib: {end-start} seconds")
        logger.info(f"Now recomputing scores")
        start = time.time()
        max_len = max([len(i) for i in opts])
        opts_tensor = torch.tensor([opt_i + [opt_i[-1]]*(max_len-len(opt_i)) for opt_i in opts],dtype=torch.int64)
        logger.info(f"opts tensor: {opts_tensor}; max_len {max_len}")
        
        if self.config.embedder.mode == "mem":
            cembs_to_rescore, cmasks_to_rescore = self.embedder.get_corpus(opts_tensor)
        else: # mode = disk
            corpus = self.embedder.get_corpus(opts_tensor)
            logger.info("All required documents are loaded")
            cembs_to_rescore = []
            cmasks_to_rescore = []
            for q_id in tqdm(range(self.embedder.qembs.shape[0]), desc="Processing queries"):
                cemb, cmask = corpus[
                    (q_id * torch.ones(opts_tensor.shape[1], dtype=torch.long, device=corpus.device)),
                    torch.arange(opts_tensor.shape[1], dtype=torch.long, device=corpus.device)
                ]

                cembs_to_rescore.append(cemb)
                cmasks_to_rescore.append(cmask)
            cembs_to_rescore = torch.stack(cembs_to_rescore)
            cmasks_to_rescore = torch.stack(cmasks_to_rescore)

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