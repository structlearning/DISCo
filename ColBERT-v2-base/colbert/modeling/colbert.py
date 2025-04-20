from colbert.infra.config.config import ColBERTConfig
from colbert.search.strided_tensor import StridedTensor
from colbert.utils.utils import print_message, flatten
from colbert.modeling.base_colbert import BaseColBERT

import torch
import string
from math import sqrt
import os


class ColBERT(BaseColBERT):
    """
        This class handles the basic encoding and scoring operations in ColBERT. It is used for training.
    """

    def __init__(self, name='bert-base-uncased', colbert_config=None):
        super().__init__(name, colbert_config)

        if self.colbert_config.mask_punctuation:
            self.skiplist = {w: True
                             for symbol in string.punctuation
                             for w in [symbol, self.raw_tokenizer.encode(symbol, add_special_tokens=False)[0]]}

    def forward(self, Q, D):
        Q = self.query(*Q)
        D, D_mask = self.doc(*D, keep_dims='return_mask')

        # Repeat each query encoding for every corresponding document.
        Q_duplicated = Q.repeat_interleave(self.colbert_config.nway, dim=0).contiguous()
        scores = self.score(Q_duplicated, D, D_mask)

        if self.colbert_config.use_ib_negatives:
            ib_loss = self.compute_ib_loss(Q, D, D_mask)
            return scores, ib_loss

        return scores

    def compute_ib_loss(self, Q, D, D_mask):
        # TODO: Organize the code below! Quite messy.
        scores = (D.unsqueeze(0) @ Q.permute(0, 2, 1).unsqueeze(1)).flatten(0, 1)  # query-major unsqueeze

        scores = colbert_score_reduce(scores, D_mask.repeat(Q.size(0), 1, 1), self.colbert_config)

        nway = self.colbert_config.nway
        all_except_self_negatives = [list(range(qidx*D.size(0), qidx*D.size(0) + nway*qidx+1)) +
                                     list(range(qidx*D.size(0) + nway * (qidx+1), qidx*D.size(0) + D.size(0)))
                                     for qidx in range(Q.size(0))]

        scores = scores[flatten(all_except_self_negatives)]
        scores = scores.view(Q.size(0), -1)  # D.size(0) - self.colbert_config.nway + 1)

        labels = torch.arange(0, Q.size(0), device=scores.device) * (self.colbert_config.nway)
        
        return torch.nn.CrossEntropyLoss()(scores, labels)

    def query(self, input_ids, attention_mask):
        input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
        Q = self.bert(input_ids, attention_mask=attention_mask)[0]
        Q = self.linear(Q)

        mask = torch.tensor(self.mask(input_ids, skiplist=[]), device=self.device).unsqueeze(2).float()
        Q = Q * mask

        return torch.nn.functional.normalize(Q, p=2, dim=2)
    
    def query_modified(self, input_ids, attention_mask):
        input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
        Q_0 = self.bert(input_ids, attention_mask=attention_mask)[0]
        Q_0 = self.linear(Q_0)
        
        mask = torch.tensor(self.mask(input_ids, skiplist=self.skiplist), device=self.device).unsqueeze(2).float()
        Q_0 = Q_0 * mask
        
        Q_dump = Q_0.clone()
        
        Q_0[:,:,-1] = 0
        Q_0 = torch.nn.functional.normalize(Q_0, p=2, dim=2)

        return Q_dump, Q_0, mask.squeeze().bool()

    def doc(self, input_ids, attention_mask, keep_dims=True):
        assert keep_dims in [True, False, 'return_mask']

        input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
        D = self.bert(input_ids, attention_mask=attention_mask)[0]
        D = self.linear(D)

        mask = torch.tensor(self.mask(input_ids, skiplist=self.skiplist), device=self.device).unsqueeze(2).float()
        D = D * mask

        D = torch.nn.functional.normalize(D, p=2, dim=2).half()

        if keep_dims is False:
            D, mask = D.cpu(), mask.bool().cpu().squeeze(-1)
            D = [d[mask[idx]] for idx, d in enumerate(D)]

        elif keep_dims == 'return_mask':
            return D, mask.bool()

        return D
    
    def doc_modified(self, input_ids, attention_mask, keep_dims=True):
        assert keep_dims in [True, False, 'return_mask']

        input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
        D = self.bert(input_ids, attention_mask=attention_mask)[0]
        D = self.linear(D)
        mask = torch.tensor(self.mask(input_ids, skiplist=self.skiplist), device=self.device).unsqueeze(2).float()
        D = D * mask
        
        if self.colbert_config.augment:
            D = self._RH_augmentation_corpus(D)
        else:
            D = self._no_augmentation(D)

        D = torch.nn.functional.normalize(D, p=2, dim=2)
        if self.use_gpu:
            D = D.half()

        if keep_dims is False:
            D, mask = D.cpu(), mask.bool().cpu().squeeze(-1)
            D = [d[mask[idx]] for idx, d in enumerate(D)]

        elif keep_dims == 'return_mask':
            return D, mask.bool()

        return D
    
    def _no_augmentation(self,embs):
        embs[:,:,-1] = 0
        return embs
    def _RH_augmentation_query(self,embs):
        # embs[:,:,-1] = -1
        if self.RH is None:
            filename = self.colbert_config.RH_file
            generate_new_rh = self.colbert_config.generate_new_rh
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
                self.RH = torch.load(filename)
        reflect = ((embs @ self.RH) >0).to(embs.dtype).unsqueeze(-1)*embs
        embs = torch.cat([embs, reflect], dim=-1)
        return embs
    def _RH_augmentation_corpus(self,embs):
        # NOTE : INDRA
        if self.colbert_config.dbl_norm:
            embs[:,:,-1] = 0
            embs = torch.nn.functional.normalize(embs, p=2, dim=2)
        
        embs[:,:,-1] = -1
        if self.RH is None:
            filename = self.colbert_config.RH_file
            generate_new_rh = self.colbert_config.generate_new_rh
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
                self.RH = torch.load(filename)
                
                
        reflect = ((embs @ self.RH) >0).to(embs.dtype).unsqueeze(-1)*embs
        embs = torch.cat([embs, reflect], dim=-1)
        return embs

    def score(self, Q, D_padded, D_mask):
        # assert self.colbert_config.similarity == 'cosine'

        if self.colbert_config.similarity == 'l2':
            assert self.colbert_config.interaction == 'colbert'
            return (-1.0 * ((Q.unsqueeze(2) - D_padded.unsqueeze(1))**2).sum(-1)).max(-1).values.sum(-1)

        return colbert_score(Q, D_padded, D_mask, config=self.colbert_config)

    def mask(self, input_ids, skiplist):
        mask = [[(x not in skiplist) and (x != 0) for x in d] for d in input_ids.cpu().tolist()]
        return mask


# TODO: In Query/DocTokenizer, use colbert.raw_tokenizer

# TODO: The masking below might also be applicable in the kNN part
def colbert_score_reduce(scores_padded, D_mask, config: ColBERTConfig):
    D_padding = ~D_mask.view(scores_padded.size(0), scores_padded.size(1)).bool()
    scores_padded[D_padding] = -9999
    scores = scores_padded.max(1).values

    assert config.interaction in ['colbert', 'flipr'], config.interaction

    if config.interaction == 'flipr':
        assert config.query_maxlen == 64, ("for now", config)
        # assert scores.size(1) == config.query_maxlen, scores.size()

        K1 = config.query_maxlen // 2
        K2 = 8

        A = scores[:, :config.query_maxlen].topk(K1, dim=-1).values.sum(-1)
        B = 0

        if K2 <= scores.size(1) - config.query_maxlen:
            B = scores[:, config.query_maxlen:].topk(K2, dim=-1).values.sum(1)

        return A + B

    return scores.sum(-1)


# TODO: Wherever this is called, pass `config=`
def colbert_score(Q, D_padded, D_mask, config=ColBERTConfig()):
    """
        Supply sizes Q = (1 | num_docs, *, dim) and D = (num_docs, *, dim).
        If Q.size(0) is 1, the matrix will be compared with all passages.
        Otherwise, each query matrix will be compared against the *aligned* passage.

        EVENTUALLY: Consider masking with -inf for the maxsim (or enforcing a ReLU).
    """

    Q, D_padded, D_mask = Q.cuda(), D_padded.cuda(), D_mask.cuda()

    assert Q.dim() == 3, Q.size()
    assert D_padded.dim() == 3, D_padded.size()
    assert Q.size(0) in [1, D_padded.size(0)]

    scores = D_padded @ Q.to(dtype=D_padded.dtype).permute(0, 2, 1)

    return colbert_score_reduce(scores, D_mask, config)


def colbert_score_packed(Q, D_packed, D_lengths, config=ColBERTConfig()):
    """
        Works with a single query only.
    """

    Q, D_packed, D_lengths = Q.cuda(), D_packed.cuda(), D_lengths.cuda()

    Q = Q.squeeze(0)

    assert Q.dim() == 2, Q.size()
    assert D_packed.dim() == 2, D_packed.size()

    scores = D_packed @ Q.to(dtype=D_packed.dtype).T

    scores_padded, scores_mask = StridedTensor(scores, D_lengths).as_padded_tensor()

    return colbert_score_reduce(scores_padded, scores_mask, config)

def colbert_score_modified(Q, D_padded, D_mask, opt_vec,  config=ColBERTConfig()):
    """
        Supply sizes Q = (1 | num_docs, *, dim) and D = (num_docs, *, dim).
        If Q.size(0) is 1, the matrix will be compared with all passages.
        Otherwise, each query matrix will be compared against the *aligned* passage.

        EVENTUALLY: Consider masking with -inf for the maxsim (or enforcing a ReLU).
    """

    Q, D_padded, D_mask = Q.cuda(), D_padded.cuda(), D_mask.cuda()

    assert Q.dim() == 3, Q.size()
    assert D_padded.dim() == 3, D_padded.size()
    assert Q.size(0) in [1, D_padded.size(0)]

    scores = D_padded @ Q.to(dtype=D_padded.dtype).permute(0, 2, 1)
    
    scores = torch.maximum(scores, opt_vec.to(scores.device))
    
    return colbert_score_reduce(scores, D_mask, config)

def colbert_score_packed_modified(Q, D_packed, D_lengths, opt_vec, config=ColBERTConfig()):
    """
        Works with a single query only.
    """

    Q, D_packed, D_lengths = Q.cuda(), D_packed.cuda(), D_lengths.cuda()

    Q = Q.squeeze(0)

    assert Q.dim() == 2, Q.size()
    assert D_packed.dim() == 2, D_packed.size()

    scores = D_packed @ Q.to(dtype=D_packed.dtype).T

    scores = torch.maximum(scores, opt_vec.to(scores.device))

    scores_padded, scores_mask = StridedTensor(scores, D_lengths).as_padded_tensor()

    return colbert_score_reduce(scores_padded, scores_mask, config)
