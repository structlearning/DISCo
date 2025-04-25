import os
import torch

from tqdm import tqdm
from typing import Union

from colbert_base.data import Collection, Queries, Ranking

from colbert_base.modeling.checkpoint import Checkpoint
from colbert_base.search.index_storage import IndexScorer

from colbert_base.infra.provenance import Provenance
from colbert_base.infra.run import Run
from colbert_base.infra.config import ColBERTConfig, RunConfig
from colbert_base.infra.launcher import print_memory_stats

TextQueries = Union[str, 'list[str]', 'dict[int, str]', Queries]


class Searcher:
    def __init__(self, index, checkpoint=None, collection=None, config=None):
        print_memory_stats()

        initial_config = ColBERTConfig.from_existing(config, Run().config)

        default_index_root = initial_config.index_root_
        self.index = os.path.join(default_index_root, index)
        self.index_config = ColBERTConfig.load_from_index(self.index)

        self.checkpoint = checkpoint or self.index_config.checkpoint
        self.checkpoint_config = ColBERTConfig.load_from_checkpoint(self.checkpoint)
        self.config = ColBERTConfig.from_existing(self.checkpoint_config, self.index_config, initial_config)

        self.collection = Collection.cast(collection or self.config.collection)
        self.configure(checkpoint=self.checkpoint, collection=self.collection)

        self.checkpoint = Checkpoint(self.checkpoint, colbert_config=self.config).cuda()
        self.ranker = IndexScorer(self.index)

        print_memory_stats()

    def configure(self, **kw_args):
        self.config.configure(**kw_args)

    def encode(self, text: TextQueries):
        queries = text if type(text) is list else [text]
        bsize = 128 if len(queries) > 128 else None

        self.checkpoint.query_tokenizer.query_maxlen = self.config.query_maxlen
        Q = self.checkpoint.queryFromText(queries, bsize=bsize, to_cpu=True)

        return Q

    def search(self, text: str, k=10):
        return self.dense_search(self.encode(text), k)

    def search_all(self, queries: TextQueries, k=10):
        queries = Queries.cast(queries)
        queries_ = list(queries.values())

        Q = self.encode(queries_)

        return self._search_all_Q(queries, Q, k)

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
        

    def _search_all_Q(self, queries, Q, k):
        all_scored_pids = [list(zip(*self.dense_search(Q[query_idx:query_idx+1], k=k)))
                           for query_idx in tqdm(range(Q.size(0)))]

        data = {qid: val for qid, val in zip(queries.keys(), all_scored_pids)}

        provenance = Provenance()
        provenance.source = 'Searcher::search_all'
        provenance.queries = queries.provenance()
        provenance.config = self.config.export()
        provenance.k = k

        return Ranking(data=data, provenance=provenance)

    def dense_search(self, Q: torch.Tensor, k=10):
        pids, scores = self.ranker.rank(self.config, Q, k)

        return pids[:k], list(range(1, k+1)), scores[:k]

    def gen_candidates(self, Q: torch.Tensor, k=10, prune_candidates=False):
        Q_aug = self.checkpoint._RH_augmentation_query(Q[:, :self.config.query_maxlen])
        Q_aug = torch.nn.functional.normalize(Q_aug.squeeze(0), dim=-1, p=2) # NOTE: Candidate generation uses only the query tokens
        if True: # self.ranker.use_gpu:
            Q_aug = Q_aug.cuda().half()
        return self.ranker.generate_candidate_pids(Q_aug, self.config.nprobe, pid_centroid_scores=prune_candidates)
    
    ## TODO: modified
    def rank_modified(self, Q, opt_vec, filter_fn=None, pids=None):
        assert pids is not None, "pids should not be None in rank_modified"
        with torch.inference_mode():
            
            pids = torch.tensor(pids, dtype=torch.int32, device=Q.device)
            centroid_scores = None

            scores = self.ranker.score_pids_modified(self.config, Q, pids, centroid_scores,opt_vec)

            scores_sorter = scores.sort(descending=True)
            pids, scores = pids[scores_sorter.indices].tolist(), scores_sorter.values.tolist()

            return pids, scores