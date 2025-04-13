import torch
import os
import numpy as np

from string import punctuation

import torch.nn.functional as F

from transformers import BertTokenizer, BertModel
import logging
from tqdm import tqdm
from .utils import load, save, batch_tensordict, batch_iterator
import torch.nn.functional as F


from colbert.infra.config import ColBERTConfig, RunConfig
from colbert import Checkpoint
from colbert.infra.run import Run

logger = logging.getLogger(__name__)



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
            
    ## IMPORTANT: 
    ## TODO: this is not tensorised. the disk op is not tensorisable, but what about the rest
    ## Indices is a tensor of indexes (int)
    def get_corpus(self,indices:torch.Tensor,padding=330):
        """
            returns corpus embeddings with masks
        """
        
            ## fetch from disk ## TODO:error found
        if self.config.mode == "disk":
            flat_indices = indices.flatten()
            assert(os.path.exists(self.embedding_path + "corpus_0.pkl"))

            raise ValueError("Bug detected in the code: run in mem mode")
            batch_size = self.config.batch_size
            
            # logger.debug(f"indices shape: {indices.shape}, flat indices shape:{flat_indices.shape}")
            unique_indices, inverse_indices = torch.unique(flat_indices, return_inverse=True)
            # logger.debug(f"unique indices shape: {unique_indices.shape}")
            # Step 2: Split indices into file numbers and sub-indices
            file_numbers = unique_indices // batch_size
            sub_indices = unique_indices % batch_size
            # logger.debug(f"file numbers shape: {file_numbers.shape}, sub indices shape: {sub_indices.shape}")
            # Step 3: Group sub-indices by file number
            file_groups = {}
            for file_num, sub_idx in zip(file_numbers.tolist(), sub_indices.tolist()):
                if file_num not in file_groups:
                    file_groups[file_num] = []
                file_groups[file_num].append(sub_idx)

            # Step 4: Load and process files
            all_cembs = []
            all_cmasks = []
            
            for file_num, sub_idxs in file_groups.items():
                # Load batch file once per group
                cemb = load(self.embedding_path + f"corpus_{file_num}.pkl")
                cmask = load(self.embedding_path + f"corpus_masks_{file_num}.pkl")
                # logger.debug(f"Loaded embeddings of shape {cemb.shape} and masks of shape {cmask.shape}")
                pad_size = padding - cemb.size(1)
                # Convert subindices to tensor and ensure valid range
                sub_idxs_tensor = torch.tensor(sub_idxs, device=cemb.device)
                sub_idxs_tensor = sub_idxs_tensor[sub_idxs_tensor < cemb.size(0)]
                # logger.debug(f"Sub indices tensor shape: {sub_idxs_tensor.shape}")
                # Gather embeddings and masks
                logger.info(f"Pad size: {pad_size}")
                all_cembs.append(F.pad(cemb[sub_idxs_tensor],(0,pad_size,0,0), mode='constant', value=0))
                all_cmasks.append(F.pad(cmask[sub_idxs_tensor],(0,pad_size), mode='constant', value=0))
                # logger.debug(f"Appended embeddings of shape {cemb[sub_idxs_tensor].shape} and masks of shape {cmask[sub_idxs_tensor].shape}")
            # Step 5: Combine results
            logger.info(f"Combining results: len(all_cembs): {len(all_cembs)}")
            logger.info(f"{all_cembs[0].shape}, {all_cembs[1].shape}")
            unique_cembs = torch.cat(all_cembs, dim=0)
            unique_cembs.div_(torch.norm(unique_cembs,dim=-1,keepdim=True))
            unique_cmasks = torch.cat(all_cmasks, dim=0)
            # logger.debug(f"Unique embeddings shape: {unique_cembs.shape}, Unique masks shape: {unique_cmasks.shape}")
            # Step 6: Map back to original indices
            final_cembs = unique_cembs[inverse_indices]
            final_cmasks = unique_cmasks[inverse_indices]
            # logger.debug(f"Final embeddings shape: {final_cembs.shape}, Final masks shape: {final_cmasks.shape}")
            
            return final_cembs.reshape(indices.shape+final_cembs.shape[1:]), final_cmasks.reshape(indices.shape+final_cmasks.shape[1:])
            
        # self.config.mode == "mem"
        return self.cembs[indices],self.cmasks[indices]
    def get_query(self,indices):
        """
            returns query embeddings with masks
        """
        return self.qembs[indices],self.qmasks[indices]
    
    ## Here it loads the embeddings from file/ generates the embeddings if required. embeddings are nt=ot generated in the ColBERTEMbedder subclass 
    def embed_full_dataset(self, data, mode="disk",pad=330):
        raise ValueError("This function should not bo longer be used.")
        self.embedding_path = f"./pickles/embeddings/{self.variety}_{self.model_name}_{data.dataset_name}/"
        if mode == "disk":
            assert(self.config.save_embeddings)
        assert mode in ["disk","mem"]
            
        logger.info(f"Embedding path: {self.embedding_path}")
        logger.info(f"Embedding mode: {mode}")
        # if embeddings already exist load them
        if (not self.config.recompute_emb) and os.path.exists(self.embedding_path+"query.pkl"):
            logger.info("Existing embeddings will be loaded")
            self.qembs = load(self.embedding_path+"query.pkl")
            self.qembs.div_(torch.norm(self.qembs,dim=-1,keepdim=True))
            self.qmasks = load(self.embedding_path+"query_masks.pkl")
            logger.info("Query embeddings loaded")
            if mode=="mem":
                
                num_batches = 0
                while os.path.exists(self.embedding_path+f"corpus_{num_batches}.pkl"):
                    num_batches+=1
                batch_size, _, __ = load(self.embedding_path+"corpus_0.pkl").shape
                total_size = (num_batches-1)*batch_size + load(self.embedding_path+f"corpus_{num_batches-1}.pkl").size(0) 
                # dtype = torch.bfloat16 ## DYTPE OF FLOAT
                dtype = torch.float32
                logger.info(f"Allocating memory for full corpus")
                self.cembs = torch.zeros((total_size,pad,self.config.emb_dim),dtype=dtype, device="cpu", pin_memory=True)
                logger.info(f"Allocating memory for full masks")
                self.cmasks = torch.zeros((total_size,pad),dtype=torch.int64, device="cpu", pin_memory=True)
                
                for i in range(num_batches):
                    cemb = load(self.embedding_path+f"corpus_{i}.pkl")
                    cmask = load(self.embedding_path+f"corpus_masks_{i}.pkl")
                    cemb.div_(torch.norm(cemb,dim=-1,keepdim=True))
                    # logger.debug(f"cemb device: {cemb.device}")
                    self.cembs[i*batch_size:i*batch_size+cemb.size(0),:cemb.size(1),:].copy_(cemb,non_blocking=True)
                    self.cmasks[i*batch_size:i*batch_size+cemb.size(0),:cemb.size(1)].copy_(cmask,non_blocking=True)
                torch.cuda.synchronize() # just in case
                logger.info("Corpus embeddings loaded")
            return
        
        # else compute embeddings
        logger.info("Computing embeddings")
        corpus, query = data.get_data()
        batch_size = self.config.batch_size
        cvals = map(lambda v : (v.get("title","") + " " + v["text"]).strip()  ,corpus.values())
        cvals_batched = tqdm(batch_iterator(cvals, batch_size=batch_size), desc="Processing Corpus", total=(len(corpus.keys())+ batch_size - 1)//batch_size)
        
        if self.compress:
            compression_matrix = self.get_compression_matrix()
        else:
            compression_matrix = torch.eye(768).unsqueeze(0).to(self.device)
        
        # todo: batch this
        logging.info("Starting with Tokenisation")
        qtokens = self.tokenizer(list(query.values()), return_tensors='pt', padding=True, truncation=True)
        logging.info("Query tokenisation complete")
        # logging.info("Corpus Tokenisation complete")
        
        logging.info("Starting with Embedding")
        with torch.no_grad(): 
            query_embs = torch.concatenate([(self.model(**qt).last_hidden_state@compression_matrix).cpu() for qt in batch_tensordict(qtokens,batch_size=batch_size,device=self.device)],dim=0)
            query_embs.div_(torch.norm(query_embs,dim=-1,keepdim=True))
            logging.info("Query Embedding complete")
            all_cembs = []
            all_cmasks = []
            for cval_batch in cvals_batched:
                tokens = self.tokenizer(list(cval_batch), return_tensors='pt', padding=True, truncation=True).to(self.device)
                embeddings = (self.model(**tokens).last_hidden_state @ compression_matrix)
                embeddings.div_(torch.norm(embeddings,dim=-1,keepdim=True))
                masks = self._mask(tokens["input_ids"], self.skiplist).to(tokens["attention_mask"].device)
                all_cembs.append(embeddings.to(device="cpu",non_blocking=True))
                all_cmasks.append(masks)
            
            logging.info("Corpus Embedding complete")
            self.qembs, self.qmasks = query_embs, qtokens["attention_mask"]
            torch.cuda.synchronize()
            if self.config.save_embeddings:
                for i,(c,m) in enumerate(zip(all_cembs, all_cmasks)):
                    save(c, f"{self.embedding_path}corpus_{i}.pkl")
                    save(m, f"{self.embedding_path}corpus_masks_{i}.pkl")
                ## change this
                save(self.qmasks, f"{self.embedding_path}query_masks.pkl") 
                save(self.qembs, f"{self.embedding_path}query.pkl")
                logger.info("Embeddings saved")
            if mode == "disk":
                del all_cembs,all_cmasks
            elif mode == "mem":
                for i in range(len(all_cembs)):
                    all_cembs[i].div_(torch.norm(all_cembs[i],dim=-1,keepdim=True))
                    all_cembs[i] = F.pad(all_cembs[i],(0,0,0,512-all_cembs[i].size(1)),mode='constant',value=0)
                    all_cmasks[i] = F.pad(all_cmasks[i],(0,512-all_cmasks[i].size(1)),mode='constant',value=0)
                self.cembs = torch.cat(all_cembs,dim=0)
                # self.cembs.div_(torch.norm(self.cembs,dim=-1,keepdim=True))
                self.cmasks = torch.cat(all_cmasks,dim=0)
            logging.info("Embedding complete")
            return
    
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
                torch.save(embs_dump, embed_dump_path+ "/all.pkl")
                torch.save(self.qmasks, mask_dump_path+"/all.pkl")
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
            logger.info("Disk mode selected, stopping here")
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
    
    


    
