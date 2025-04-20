import torch
import os
import numpy as np
import time
from string import punctuation

from collections import defaultdict

import torch.nn.functional as F

from transformers import BertTokenizer, BertModel
import logging
from tqdm import tqdm
from .utils import load, save, batch_tensordict, batch_iterator
import torch.nn.functional as F


from colbert.infra.config import ColBERTConfig, RunConfig
from colbert import Checkpoint
from colbert.infra.run import Run
import gc
logger = logging.getLogger(__name__)


class RetrievalCorpus:
    def __init__(self, unique_emb, unique_mask, idx_matrix):
        """
        unique_emb  : (U, padding, D) float tensor
        unique_mask : (U, padding) bool  tensor
        idx_matrix  : (Q, K) long  tensor
        """
        self.emb     = unique_emb
        self.mask    = unique_mask
        self.idx     = idx_matrix
        self.device = unique_emb.device

    def __getitem__(self, key):
        # Expect key to be a 2‐tuple: (q_idx, k_idx)
        if not (isinstance(key, tuple) and len(key) == 2):
            raise IndexError("Use corpus[q_idx, k_idx]")

        q_idx, k_idx = key
        # this handles ints, slices, LongTensors, bool masks, etc.
        u_idx = self.idx[q_idx, k_idx]  # shape = broadcast(q_idx,k_idx)

        # now emb[u_idx] has shape = (*u_idx.shape, pad, D)
        return self.emb[u_idx], self.mask[u_idx]

    def full(self):
        # return the full (Q,K,pad,D) and (Q,K,pad)
        return self.emb[self.idx], self.mask[self.idx]

# parts of this code are modified from the beir lib
# A bert embedder with random {-1,1} compression.
# this class has been superceded by the ColBERTEmbedder. However take a look at the inherited methods
# The embedder handles the embeddings that can will be used for reranking (random access) and iterate in batches over query/corpus
# it has two modes, an in-memory "mem" mode, and a "disk" mode
# random access is not correctly implemented for disk mode
# Maybe we should put this to run in a database for random access
class BERTEmbedder:
    def __init__(self, config):
        # self.variety = "bert"
        self.config = config
        try:
            self.pretrained = config.pretrained
            self.model_name = config.model_name
        except:
            self.pretrained = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Embedder using device: {self.device}")
        
        # self.compress = self.config.emb_dim != 768
        # if self.compress:
        #     self.variety += f"_d{self.config.emb_dim}" 
        
        
        if self.pretrained:
            logger.info(f"Loading pretrained model: {self.model_name}")
            self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
            self.model = BertModel.from_pretrained(self.model_name)
            self.skiplist = {w: True
                             for symbol in punctuation
                             for w in [symbol, self.tokenizer.encode(symbol, add_special_tokens=False)[0]]}
        
            self.model = torch.nn.DataParallel(self.model)
            self.model.to(self.device)
            self.model.eval()
        
            self.embedding_path = None
        
        self.qtokens = None
        self.ctokens = None
        self.cembs = None
        self.qembs = None
        self.cmasks = None
        self.qmasks = None
        
    def _mask(self,tokens,skiplist):
        mask = [[(x not in skiplist) and (x != self.pad_token) for x in d] for d in tokens.cpu().tolist()]
        return torch.tensor(mask)
        
    # def get_compression_matrix(self):
    #     compression_matrix_path = f"./pickles/embeddings/compression_matrix_{self.variety}_{self.model_name}.pt"
    #     if os.path.exists(compression_matrix_path):
    #         compression_matrix = load(compression_matrix_path,device=self.device)
    #     else:
    #         compression_matrix = (torch.randint(0, 2,(1,768,self.config.emb_dim),dtype=torch.float32) * 2 - 1).to(self.device)
    #         save(compression_matrix,compression_matrix_path)
    #     return compression_matrix
    
    def get_corpus_by_batch(self,device,start=0):
        batch_size = self.config.batch_size
        if self.cembs is None: # fetch from disk
            assert os.path.exists(self.embedding_path+"corpus_0.pkl")
            
            i=0
            while os.path.exists(self.embedding_path+f"corpus_{i}.pkl"):
                embs = load(self.embedding_path+f"corpus_{i}.pkl").to(device)
                embs.div_(torch.norm(embs,dim=-1,keepdim=True))
                yield embs,load(self.embedding_path+f"corpus_masks_{i}.pkl").to(device)
                i+=1
            
        else:
            for i in range(start,self.cembs.size(0),batch_size):
                yield self.cembs[i:i+batch_size].to(device),self.cmasks[i:i+batch_size].to(device)
    def get_query_batched(self,batch_size,device,start=0):
        for i in range(start,self.qembs.size(0), batch_size):
            yield self.qembs[i:i+batch_size].to(device),self.qmasks[i:i+batch_size].to(device)


    def create_mapping(self, indices):
        mapping = {}
        H, W = indices.shape

        for i in range(H):
            for j in range(W):
                batch_info = self.docid_to_batchinfo[str(indices[i][j].item())]
                b   = batch_info[0]
                m   = batch_info[1]
                off = batch_info[2]
                key = (b, m)
                # store offset and the 2D index tuple
                mapping.setdefault(key, []).append((off, (i, j)))

        # sort each list by offset
        for key in mapping:
            mapping[key].sort(key=lambda x: x[0])

        return mapping
        
    ## IMPORTANT: 
    ## TODO: this is not tensorised. the disk op is not tensorisable, but what about the rest
    ## Indices is a tensor of indexes (int)
    def get_corpus(self,indices:torch.Tensor,padding=330):
        """
            returns corpus embeddings with masks
        """
        
            ## fetch from disk ## TODO:error found
        if self.config.mode == "disk":
            # indices: (Q, K) global doc‐IDs
            Q, K = indices.shape
            flat = indices.view(-1)  # (Q*K,)

            # find the unique docs and build a back‐pointer
            unique_ids, inv = torch.unique(flat, sorted=True, return_inverse=True)
            # inv: (Q*K,) tells you for each flattened slot which unique‐row to use
            inv = inv.view(Q, K)  # now (Q, K)

            U = unique_ids.size(0)
            D = self.config.emb_dim

            # preallocate the unique buffers
            device = indices.device
            unique_emb  = torch.zeros(U, padding, D, device=device)
            unique_mask = torch.zeros(U, padding, dtype=torch.bool, device=device)

            # group by minibatch - so that each .pkl file is loaded only once
            shard_groups = defaultdict(list)
            # mapping doc_id -> (batch, shard, local_offset)
            for u_idx, doc_id in enumerate(unique_ids.tolist()):
                b, s, off = self.docid_to_batchinfo[str(doc_id)]
                shard_groups[(b, s)].append((u_idx, off))

            # 4) for each shard: load once, pad once, then pick out all needed rows
            for (b, s), lst in shard_groups.items():
                print(f"Batch {b} Minibatch {s}")
                fp   = self.embedding_path(f"compressed_{D}", b, s)
                data = torch.load(fp, map_location='cpu')
                cemb = data['embs_compressed']   # (N, L, D)
                cm   = torch.load(self.embedding_path("masks", b, s),
                                map_location='cpu')['masks']  # (N, L)
                # zero‑out last dim
                cemb[..., -1] = 0

                # pad to padding set size
                L = cemb.shape[1]
                if L < padding:
                    pad_len = padding - L
                    cemb = torch.cat([cemb, torch.zeros((cemb.size(0), pad_len, D), dtype=cemb.dtype, device='cpu')], dim=1)
                    cm = torch.cat([cm, torch.zeros((cm.size(0), pad_len), dtype=cm.dtype, device='cpu')], dim=1)
                cemb = cemb.to(device)
                cm = cm.to(device)

                # unpack (u_idx, off) list
                u_idxs, offs = zip(*lst)
                u_t   = torch.tensor(u_idxs, dtype=torch.long, device=device)
                off_t = torch.tensor( offs, dtype=torch.long, device=device)

                # fill unique buffers
                unique_emb[u_t] = cemb[off_t]
                unique_mask[u_t] = cm[off_t]

                del cemb, cm, data
                torch.cuda.empty_cache()

            # Normalize all embeddings to unit‑norm
            unique_emb = F.normalize(unique_emb, p=2, dim=-1)
            return RetrievalCorpus(unique_emb, unique_mask, inv)
        
        return self.cembs[indices],self.cmasks[indices]
    def get_query(self,indices):
        """
            returns query embeddings with masks
        """
        return self.qembs[indices],self.qmasks[indices]
    def print_dataset_statistics(self):
        pass 
    
## A handler for embeddings generated by colbert
## This is going to be used everywhere
class ColBERTEmbedder(BERTEmbedder):
    def __init__(self, config):
        super().__init__(config)
        self.type = config.type
        self.mv_type = config.mv_type
        # self.variety = "colbert"
        # self.compress = self.config.emb_dim != 768
        self.num_batches = None ## the batch size is determined by the files present, no changes are being made here
        # if self.compress:
        #     self.variety += f"_d{self.config.emb_dim}" 
        # logger.info(f"ColBERT variety: {self.variety}")
    
    def embed_queries(self,query, dump_path):
        ## use self.qembs, self.qmasks to get queries
        logger.info("Embedding queries")
        query_texts = list(query.values())
        with Run().context(RunConfig(nranks=1, experiment=f"{self.dataset_name}")):
            #NOTE: AUGMENTATION IS OFF 
            config = ColBERTConfig(root="./colbert_beir_expts/", lin_dim=self.config.emb_dim)
            # config.generate_new_rh = self.config.generate_new_rh
            # config.RH_file = self.config.RH_file
            colbert_model = Checkpoint(colbert_config=config, name="ColBERT/colbertv2.0")
            with torch.inference_mode():
                embs_dump, self.qembs, self.qmasks = colbert_model.queryFromText_modified(query_texts, bsize=self.config.query_batch_size)
                embed_dump_path = f"{dump_path}/compressed_{self.config.emb_dim}"
                mask_dump_path = f"{dump_path}/masks"
                os.makedirs(embed_dump_path, exist_ok=True)
                os.makedirs(mask_dump_path, exist_ok=True)
                ############### Commented because I have no write access to embedding folder ############
                # torch.save(embs_dump, embed_dump_path+ "/all.pkl")
                # torch.save(self.qmasks, mask_dump_path+"/all.pkl")
                #########################################################################################
                self.qembs = torch.nn.functional.normalize(self.qembs,dim=-1,p=2)  #NOTE: query_modified already returning [:,:,-1] = 0 -> normalized embeds. Perhaps unnecessary
                self.qmasks = self.qmasks.to(self.qembs.device)

    def get_query(self,indices):
        """
            returns query embeddings with masks
        """
        return super().get_query(indices)
     
    def embed_full_dataset(self,data,mode="disk",pad=330):
        dataset_name = data.dataset_name
        self.dataset_name = dataset_name
        assert mode in ["disk","mem"]
        prefix_str = f"./experiments/{dataset_name}/{self.type}"
        self.status_file  = f"{prefix_str}/corpus/status.json"
        # self.status_file = f"./experiments/colbert_{dataset_name}/indexes/{dataset_name}_embs/status.json"
        self.embedding_path = lambda folder,batch,minibatch : f"{prefix_str}/corpus/{folder}/batch_{batch}.{minibatch}.pkl"
        
        _, query = data.get_data()
        self.embed_queries(query, f"{prefix_str}/query")
        ## sanity checks on file
        status_json = load(self.status_file)
        self.num_batches = len(status_json) // 2  # We are also storing the number of mini-batches ber batch
        for i in range(self.num_batches):
            assert status_json.get(f"status.{i}",False), f"Status file indicates encoding is not completed for batch {i}"
            print(self.embedding_path(f"compressed_{self.config.emb_dim}",i,0))
            assert os.path.exists(self.embedding_path(f"compressed_{self.config.emb_dim}",i,0)), f"Compressed file does not exist for batch {i}"
            assert os.path.exists(self.embedding_path("masks",i,0)), f"Mask file does not exist for batch {i}"
        
        ## if disk mode return
        if mode=="disk":
            self.docid_to_batchinfo_file  = f"./docid_to_batchinfo/docid_to_batchinfo_{self.dataset_name}.json"
            self.docid_to_batchinfo = load(self.docid_to_batchinfo_file)
            return
        ## if mem mode load to memory
        assert mode == "mem", "Mode should be mem"
        logger.info("Loading embeddings to memory")
        cembs_list = []
        cmasks_list = []
        max_len = 0
        num = 0
        for i in range(self.num_batches):
            j = 0
            while os.path.exists(self.embedding_path(f"compressed_{self.config.emb_dim}", i, j)):
                cemb = torch.load(self.embedding_path(f"compressed_{self.config.emb_dim}", i, j))["embs_compressed"]
                cmask = torch.load(self.embedding_path("masks", i, j))["masks"]
                cemb[:,:,-1] = 0 #Important
                cembs_list.append(cemb)
                cmasks_list.append(cmask)
                j += 1
                num += cmask.size(0)
                max_len = max(max_len,cmask.size(1))
                print(i,j,"loaded")

        logger.info(f"Loaded to list, fitting to tensor")
        self.cembs = torch.zeros((num,max_len,self.config.emb_dim),dtype=torch.float32,device="cpu",pin_memory=True)
        self.cmasks = torch.zeros((num,max_len),dtype=torch.float32,device="cpu",pin_memory=True)
        start = 0
        for i in range(len(cembs_list)):
            end = start + cmasks_list[i].size(0)
            self.cembs[start:end,:cmasks_list[i].size(1),:] = cembs_list[i]
            self.cmasks[start:end,:cmasks_list[i].size(1)] = cmasks_list[i].to(self.cmasks.dtype)
            start = end
        
        self.cembs = torch.nn.functional.normalize(self.cembs,dim=-1,p=2) 
        
                
    def get_corpus(self, indices, padding=330):
        return super().get_corpus(indices, padding)
    def get_corpus_by_batch(self, device, start=0):
        return super().get_corpus_by_batch(device, start)
    
    


    
