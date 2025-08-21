import os
import torch

from tqdm import tqdm
from typing import Union

from colbert.data import Collection, Queries, Ranking

from colbert.modeling.checkpoint import Checkpoint
from colbert.search.index_storage import IndexScorer

from colbert.infra.provenance import Provenance
from colbert.infra.run import Run
from colbert.infra.config import ColBERTConfig, RunConfig
from colbert.infra.launcher import print_memory_stats

from math import ceil
from colbert.modeling.colbert import colbert_score_reduce
from colbert.search.strided_tensor import StridedTensor

import time

TextQueries = Union[str, 'list[str]', 'dict[int, str]', Queries]


class Searcher:
    def __init__(self, index, checkpoint=None, collection=None, config=None, index_root=None, verbose:int = 3):
        self.verbose = verbose
        if self.verbose > 1:
            print_memory_stats()

        initial_config = ColBERTConfig.from_existing(config, Run().config)

        default_index_root = initial_config.index_root_
        index_root = index_root if index_root else default_index_root
        self.index = os.path.join(index_root, index)
        self.index_config = ColBERTConfig.load_from_index(self.index)

        self.checkpoint = checkpoint or self.index_config.checkpoint
        self.checkpoint_config = ColBERTConfig.load_from_checkpoint(self.checkpoint)
        self.config = ColBERTConfig.from_existing(self.checkpoint_config, self.index_config, initial_config)

        self.collection = Collection.cast(collection or self.config.collection)
        self.configure(checkpoint=self.checkpoint, collection=self.collection)

        self.checkpoint = Checkpoint(self.checkpoint, colbert_config=self.config, verbose=self.verbose)
        use_gpu = self.config.total_visible_gpus > 0
        if use_gpu:
            self.checkpoint = self.checkpoint.cuda()
        load_index_with_mmap = self.config.load_index_with_mmap
        if load_index_with_mmap and use_gpu:
            raise ValueError(f"Memory-mapped index can only be used with CPU!")
        self.ranker = IndexScorer(self.index, use_gpu, load_index_with_mmap)

        print_memory_stats()

    def configure(self, **kw_args):
        self.config.configure(**kw_args)

    def encode(self, text: TextQueries, full_length_search=False):
        queries = text if type(text) is list else [text]
        bsize = 128 if len(queries) > 128 else None

        self.checkpoint.query_tokenizer.query_maxlen = self.config.query_maxlen
        Q = self.checkpoint.queryFromText(queries, bsize=bsize, to_cpu=True, full_length_search=full_length_search)

        return Q

    def search(self, text: str, k=10, filter_fn=None, full_length_search=False, pids=None):
        raise ValueError("Not compatible with our monkeypatch, as we providing embeddings. comment this out/ modify this accordingly")
        Q = self.encode(text, full_length_search=full_length_search)
        return self.densse_search(Q, k, filter_fn=filter_fn, pids=pids)

    def search_all(self, queries: TextQueries, k=10, filter_fn=None, full_length_search=False, qid_to_pids=None):
        queries = Queries.cast(queries)
        queries_ = list(queries.values())

        Q = self.encode(queries_, full_length_search=full_length_search)

        return self._search_all_Q(queries, Q, k, filter_fn=filter_fn, qid_to_pids=qid_to_pids)

    # queries; dummy class with .values() and .provenance() methods
    def search_all_modified(self, queries, Q, M, k,normalize_q=True):

        if self.config.augment: 
            Q  = self.checkpoint._RH_augmentation_query(Q)
            #NOTE: necessary because optvec is also appended to dummy 0 rows and so we need to mask it out. Similar behavior can be achieved by doinf the masking before augmentation
            Q = Q * M[:,:,None].to(torch.float32)
            if normalize_q:
                Q  = torch.nn.functional.normalize(Q, dim=-1, p=2) #NOTE: Important

        assert (torch.allclose(torch.norm(Q.to(torch.float32), p=2, dim=-1) ,  M.to(torch.float32), atol=1e-5)), "Q is not normalized, please check the code"
        
        # assert torch.allclose( torch.norm(Q.to(torch.float32), p=2, dim=-1), torch.ones((Q.size(0), Q.size(1)), device=Q.device, dtype=torch.float32), atol=1e-3), "Q is not normalized, please check the code"


        return self._search_all_Q(queries, Q, k) 
        
## original colbert
    def _search_all_Q(self, queries, Q, k, filter_fn=None, qid_to_pids=None):
        qids = list(queries.keys())

        if qid_to_pids is None:
            qid_to_pids = {qid: None for qid in qids}

        all_scored_pids = [
            list(
                zip(
                    *self.dense_search(
                        Q[query_idx:query_idx+1],
                        k, filter_fn=filter_fn,
                        pids=qid_to_pids[qid]
                    )
                )
            )
            for query_idx, qid in tqdm(enumerate(qids))
        ]

        data = {qid: val for qid, val in zip(queries.keys(), all_scored_pids)}

        provenance = Provenance()
        provenance.source = 'Searcher::search_all'
        provenance.queries = queries.provenance()
        provenance.config = self.config.export()
        provenance.k = k

        return Ranking(data=data, provenance=provenance)

    def dense_search(self, Q: torch.Tensor, k=10, filter_fn=None, pids=None):
        if k <= 10:
            if self.config.ncells is None:
                self.configure(ncells=1)
            if self.config.centroid_score_threshold is None:
                self.configure(centroid_score_threshold=0.5)
            if self.config.ndocs is None:
                self.configure(ndocs=256)
        elif k <= 100:
            if self.config.ncells is None:
                self.configure(ncells=2)
            if self.config.centroid_score_threshold is None:
                self.configure(centroid_score_threshold=0.45)
            if self.config.ndocs is None:
                self.configure(ndocs=1024)
        else:
            if self.config.ncells is None:
                self.configure(ncells=4)
            if self.config.centroid_score_threshold is None:
                self.configure(centroid_score_threshold=0.4)
            if self.config.ndocs is None:
                self.configure(ndocs=max(k * 4, 4096))

        pids, scores = self.ranker.rank(self.config, Q, filter_fn=filter_fn, pids=pids)

        return pids[:k], list(range(1, k+1)), scores[:k]

    def gen_candidates(self, Q: torch.Tensor, k=10, prune_candidates=False):
        if k <= 10:
            if self.config.ncells is None:
                self.configure(ncells=1)
            if self.config.centroid_score_threshold is None:
                self.configure(centroid_score_threshold=0.5)
            if self.config.ndocs is None:
                self.configure(ndocs=256)
        elif k <= 100:
            if self.config.ncells is None:
                self.configure(ncells=2)
            if self.config.centroid_score_threshold is None:
                self.configure(centroid_score_threshold=0.45)
            if self.config.ndocs is None:
                self.configure(ndocs=1024)
        else:
            if self.config.ncells is None:
                self.configure(ncells=4)
            if self.config.centroid_score_threshold is None:
                self.configure(centroid_score_threshold=0.4)
            if self.config.ndocs is None:
                self.configure(ndocs=max(k * 4, 4096))
        Q_aug = self.checkpoint._RH_augmentation_query(Q[:, :self.config.query_maxlen])
        Q_aug = torch.nn.functional.normalize(Q_aug.squeeze(0), dim=-1, p=2) # NOTE: Candidate generation uses only the query tokens
        if self.ranker.use_gpu:
            Q_aug = Q_aug.cuda().half()
        pids,centroid_scores = self.ranker.generate_candidate_pids(Q_aug, self.config.ncells)
        
        if not prune_candidates:
            return pids, centroid_scores, None
        
        ### the following is picked up from score_pids in search/index_storage.py
        batch_size = 2 ** 20
        idx = centroid_scores.max(-1).values >= self.config.centroid_score_threshold
        
        approx_scores = []

        # Filter docs using pruned centroid scores
        for i in range(0, ceil(len(pids) / batch_size)):
            pids_ = pids[i * batch_size : (i+1) * batch_size]
            codes_packed, codes_lengths = self.ranker.embeddings_strided.lookup_codes(pids_)
            idx_ = idx[codes_packed.long()]
            pruned_codes_strided = StridedTensor(idx_, codes_lengths, use_gpu=True)
            pruned_codes_padded, pruned_codes_mask = pruned_codes_strided.as_padded_tensor()
            pruned_codes_lengths = (pruned_codes_padded * pruned_codes_mask).sum(dim=1)
            codes_packed_ = codes_packed[idx_]
            approx_scores_ = centroid_scores[codes_packed_.long()]
            if approx_scores_.shape[0] == 0:
                approx_scores.append(torch.zeros((len(pids_),), dtype=approx_scores_.dtype).cuda())
                continue
            approx_scores_strided = StridedTensor(approx_scores_, pruned_codes_lengths, use_gpu=True)
            approx_scores_padded, approx_scores_mask = approx_scores_strided.as_padded_tensor()
            approx_scores_ = colbert_score_reduce(approx_scores_padded, approx_scores_mask, self.config)
            approx_scores.append(approx_scores_)
        approx_scores = torch.cat(approx_scores, dim=0)
        assert approx_scores.is_cuda, approx_scores.device
        if self.config.ndocs < len(approx_scores):
            pids = pids[torch.topk(approx_scores, k=self.config.ndocs).indices]
            
        ## Here we are getting the approx scores from the centroid scores that is returned ahead
        codes_packed, codes_lengths = self.ranker.embeddings_strided.lookup_codes(pids)
        approx_scores = centroid_scores[codes_packed.long()]
        return pids, approx_scores, codes_lengths
        

    def rank_modified(self, Q, opt_vec, filter_fn=None, pids=None):
        assert pids is not None, "pids should not be None in rank_modified"
        with torch.inference_mode():

            # Detach and clone pids and avoid initializing a new tensor using the pids tensor
            # This can lead to bus error problems on NFS/ZFS style filesystems
            pids = pids.detach().clone().to(dtype=torch.int32, device=Q.device)
            centroid_scores = None

            scores, pids = self.ranker.score_pids_modified(self.config, Q, pids, centroid_scores,opt_vec)

            scores_sorter = scores.sort(descending=True)
            pids, scores = pids[scores_sorter.indices].tolist(), scores_sorter.values.tolist()

            return pids, scores