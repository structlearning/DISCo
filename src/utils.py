import torch
import torch.nn.functional as F
import random
import numpy as np
import os,pickle
import logging
import json 
logger = logging.getLogger(__name__)

## Three broad classes of utilities
# 1. "partial_chamfer"
# 2. generators for batching/similar utilities
# 3. seeding, saving, loading

## What i call "Partial_Chamfer"
## Partial_Chamfer(Q,S) = [ max_{token x in all docs in S} q.x for q in Q ]. essentially what we get after summing this vector is chamfer(Q,S)
## why this is important: this is basically markov.
## (i) Partial_Chamfer(Q,S+{e}) = elementwise_max {Partial_Chamfer(Q,S), Partial_Chamfer(Q,{e}) }
## (ii) Our initial augmentation for the query (before RH aug) is basically concatenation with partial_chamfer upto now
### the functions here compute Partial_Chamfer(Q,{e}) for different settings



## TODO: change devices and other opts
def partial_chamfer_sim_batched_with_rerank(query, query_masks,running_optvec, max_gain_corpus, max_gain_corpus_masks):
    
    float32_min = -10
    
    # max_gain_corpus shape : (query_num,bucket_size,corpus_Set_Size,emb_dim)
    # max_gain_corpus_masks shape : (query_num,bucket_size,corpus_Set_Size)
    # ops to do: 
    
    # Move everything to CPU.
    device = torch.device("cuda")
    
    query = query.to(device)
    query_masks = query_masks.to(device)
    max_gain_corpus = max_gain_corpus.to(device)
    max_gain_corpus_masks = max_gain_corpus_masks.to(device)
    running_optvec = running_optvec.to(device)
    # logger.debug(f"max_gain_corpus shape : {max_gain_corpus.shape}")
    # logger.debug(f"query shap : {query.shape}")
    
    # Compute dot products between query tokens and corpus tokens.
    # batch matrix multiplication:
    # (query_num, query_set_size, emb_dim) @ (query_num, emb_dim, corpus_set_size)
    # results in a tensor of shape: (query_num, query_set_size, corpus_set_size)
    sim = torch.einsum("qbce,qse->qsbc",max_gain_corpus,query)
    # sim: query_num, query_set_size, bucket_size, corpus_set_size
    
    # Apply the query mask: expand from (query_num, query_set_size) to (query_num, query_set_size, 1,1)
    # This will zero out any dot products for invalid query tokens.
    sim_qmasked = sim * query_masks.unsqueeze(-1).unsqueeze(-1).to(dtype=sim.dtype)
    
    # For corpus tokens, we want to ignore invalid tokens.
    # Expand the corpus mask from (query_num,bucket_size, corpus_set_size) to (query_num, 1,bucket_size, corpus_set_size)
    # and then use torch.where to replace dot products for invalid corpus tokens with -infinity.
    sim_masked = torch.where(
        max_gain_corpus_masks.unsqueeze(1).bool(), 
        sim_qmasked, 
        torch.full_like(sim_qmasked, float32_min)
    )
    
    # For each query token, take the maximum dot product over corpus tokens (dimension 3).
    max_sim_0 = torch.amax(sim_masked, dim=3)*query_masks.unsqueeze(-1)  # Shape: (query_num, query_set_size,bucket_size)
    # subtract the running optimum and relu to get the marginal score 
    max_sim_0 = torch.maximum(max_sim_0, running_optvec)
    
    scores = torch.sum(max_sim_0,dim=1) # shape: (query_num,bucket_size)
    # save(scores, "debug_scores.pkl")
    
    max_sim_scores, max_sim_indices = torch.max(scores,dim=1) # shape: (query_num,)
    # max_sim = (query_num,query_set_size,bucket_size[max_sim_indices])
    # Extract the max values from max_sim_0 using max_sim_indices
    query_num, query_set_size, _ = max_sim_0.shape
    batch_indices = torch.arange(query_num).unsqueeze(1).expand(-1, query_set_size)  # Shape: (query_num, query_set_size)
    
    max_sim = max_sim_0[batch_indices, torch.arange(query_set_size).expand(query_num, -1), max_sim_indices.unsqueeze(1)]

    return max_sim, max_sim_indices, max_sim_scores


def partial_chamfer_sim_batched_with_rerank_aug(query, query_masks, running_optvec, max_gain_corpus, max_gain_corpus_masks):
    
    float32_min = -10
    device = torch.device("cuda")
    
    query = query.to(device)
    query_masks = query_masks.to(device)
    max_gain_corpus = max_gain_corpus.to(device)
    max_gain_corpus_masks = max_gain_corpus_masks.to(device)
    running_optvec = running_optvec.to(device)
    # logger.debug(f"max_gain_corpus shape : {max_gain_corpus.shape}")
    # logger.debug(f"query shap : {query.shape}")
    
    # Compute dot products between query tokens and corpus tokens.
    # batch matrix multiplication:
    sim = torch.einsum("qbcer,qser->qsbcr", max_gain_corpus, query)
    # sim: query_num, query_set_size, bucket_size, corpus_set_size
    
    # Apply the query mask: expand from (query_num, query_set_size) to (query_num, query_set_size, 1, 1, 1)
    # This will zero out any dot products for invalid query tokens.
    sim_qmasked = sim * query_masks.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(dtype=sim.dtype)
    
    # For corpus tokens, we want to ignore invalid tokens.
    # Expand the corpus mask from (query_num,bucket_size, corpus_set_size) to (query_num, 1, bucket_size, corpus_set_size, 1)
    # and then use torch.where to replace dot products for invalid corpus tokens with -infinity.
    sim_masked = torch.where(
        max_gain_corpus_masks.unsqueeze(1).unsqueeze(-1).bool(), 
        sim_qmasked, 
        torch.full_like(sim_qmasked, float32_min)
    )
    
    # For each query token, take the maximum dot product over corpus tokens (dimensions 3 and 4).
    max_sim_0 = torch.amax(sim_masked, dim=[3, 4])*query_masks.unsqueeze(-1)  # Shape: (query_num, query_set_size, bucket_size)
    # subtract the running optimum and relu to get the marginal score 
    max_sim_0 = torch.maximum(max_sim_0, running_optvec)

    scores = torch.sum(max_sim_0,dim=1) # shape: (query_num,bucket_size)
    # save(scores, "debug_scores.pkl")
    
    max_sim_scores, max_sim_indices = torch.max(scores,dim=1) # shape: (query_num,)
    # max_sim = (query_num,query_set_size,bucket_size[max_sim_indices])
    # Extract the max values from max_sim_0 using max_sim_indices
    query_num, query_set_size, _ = max_sim_0.shape
    batch_indices = torch.arange(query_num).unsqueeze(1).expand(-1, query_set_size)  # Shape: (query_num, query_set_size)
    
    max_sim = max_sim_0[batch_indices, torch.arange(query_set_size).expand(query_num, -1), max_sim_indices.unsqueeze(1)]

    return max_sim, max_sim_indices, max_sim_scores



float32_min = -10
# 1 query full corpus
#query: (query_set_size,emb_dim) ; corpus: (corpus_num, corpus_set_size, emb_dim)
def partial_chamfer_sim(query,corpus,cmasks,device=None,bs=1024):
    #output: (query_set_size,corpus_num)
    # computes the vector \max_{x \in corpus} <x,query>
    out = torch.empty(query.size(0),corpus.size(0),dtype=query.dtype)
    ## batch_size : 
    # ##can decrease/increase this depending on the footprint you have available
    if device is None:
        device = query.device
    else:
        query = query.to(device)
    for i,(corpus_batch,corpus_masks_batch) in enumerate(zip(batch_tensor(corpus,bs,device),batch_tensor(cmasks,bs,device))):
        sim = torch.matmul(corpus_batch,query.T.to(dtype=corpus_batch.dtype)).permute(2, 0, 1)
        # sim: (query_set_size,corpus_num,corpus_set_size)
        masked_sim = torch.where(corpus_masks_batch.bool().unsqueeze(0), sim, float32_min)
        out[:,i*bs:i*bs+corpus_batch.size(0)].copy_(torch.amax(masked_sim,dim=2),non_blocking=False)
    torch.cuda.synchronize()
    return out


def batch_iterator(iterator, batch_size):
    # returns a generator that returns batches of size batch_size
    batch = []
    for item in iterator:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if len(batch) > 0:
        yield batch

def batch_tensor(cpu_tensor, batch_size,device):
    # returns a generator that returns batches of size batch_size. initial tensor lies on cpu
    for i in range(0, cpu_tensor.size(0), batch_size):
        yield (cpu_tensor[i:i + batch_size]).to(device)

def batch_tensordict(cpu_tensordict, batch_size,device):
    # returns a generator that returns batches of size batch_size. initial tensor lies on cpu
    leng = cpu_tensordict[list(cpu_tensordict.keys())[0]].size(0)
    for i in range(0, leng, batch_size):
        yield {k:(v[i:i + batch_size]).to(device) for k,v in cpu_tensordict.items()}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed + 1)
    torch.manual_seed(seed + 2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.use_deterministic_algorithms(True)
    
def set_seed_from_checkpoint(random_seed,np_random_seed,torch_random_seed):
    random.seed(random_seed)
    np.random.seed(np_random_seed)
    torch.manual_seed(torch_random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.use_deterministic_algorithms(True)
    
def save(obj, path):
    
    dir_name = os.path.dirname(path)
    if dir_name: 
        os.makedirs(dir_name, exist_ok=True)
    logger.info(f"Saving object to {path}")
    if path.endswith(".pkl"):
        with open(path, "wb") as f:
            pickle.dump(obj, f)
            f.flush()
    elif path.endswith(".pt"):
        torch.save(obj, path)
    elif path.endswith(".npy"):
        np.save(path, obj)
    elif path.endswith(".json"):
        with open(path, "w") as f:
            json.dump(obj, f)
            f.flush()
    else:
        logger.error(f"Unknown file type: {path}")
        raise ValueError(f"Unknown file type: {path}")

def load(path, device="cpu"):
    logger.info(f"Loading object from {path}")
    if path.endswith(".pkl"):
        with open(path, "rb") as f:
            return pickle.load(f)
    elif path.endswith(".pt"):
        return torch.load(path, map_location=device,weights_only=True)
    elif path.endswith(".npy"):
        return np.load(path)
    elif path.endswith(".json"):
        with open(path, "r") as f:
            return json.load(f)
    else:
        logger.error(f"Unknown file type: {path}")
        raise ValueError(f"Unknown file type: {path}")


def hamming_distance(x, y):
    xor = x ^ y
    # Determine bit width dynamically (e.g., for 64-bit hashes, use 64)
    bit_width = max(xor.max().item().bit_length(), y.max().item().bit_length())
    bits = ((xor.unsqueeze(-1) >> torch.arange(bit_width, device=xor.device)) & 1)
    logger.info("Hamming distance calculated")
    return bits.sum(dim=-1)


import numpy as np

def rowwise_union_padded(arrays, num_rh, k, preserve_order=False):
    """
    arrays: list of numpy arrays, each of shape (Q, k), integer dtype
    preserve_order: if True, keep first-appearance order per row; else sorted

    Returns:
        out: (Q, max_union_len) int array,
             padded with the first element of each row
    """
    assert len(arrays) == num_rh, f"Expecting exactly {num_rh} arrays"
    Q = arrays[0].shape[0]
    for a in arrays:
        assert a.shape == (Q, k), f"All arrays must be (Q, {k})"

    # Concatenate across axis=1 -> (Q, num_rh * k)
    stacked = np.concatenate(arrays, axis=1)

    unions = []
    if preserve_order:
        for row in stacked:
            vals, idx = np.unique(row, return_index=True)
            unions.append(row[np.sort(idx)])  # keep order of first appearance
    else:
        for row in stacked:
            unions.append(np.unique(row))     # sorted

    max_len = max(len(u) for u in unions) if unions else 0

    out = np.empty((Q, max_len), dtype=stacked.dtype)
    for i, u in enumerate(unions):
        out[i, :len(u)] = u
        if len(u) < max_len:
            out[i, len(u):] = u[0]  # pad with the first element of that row

    return out