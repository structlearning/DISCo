import os
import pickle

import torch

import logging
logger = logging.getLogger(__name__)


class StateSaverBase(object):
    def __init__(self, *args, **kwargs):
        self.dataset = kwargs.get('dataset')
        super().__init__()
        self.state = {}

        self.prefix = "results/intermediate"
        os.makedirs(self.prefix, exist_ok=True)

        self.path = f"{self.prefix}/{self.dataset}"

    def serialize(self):
        """
        Ensure that the state is saved atomically.
        This way, if the process crashes while writing, the original file remains intact.
        """
        logger.info(f"Saving state to {self.tmp_path}")
        torch.save(self.state, self.tmp_path)
        logger.info(f"Moving from {self.tmp_path} to {self.path}")
        os.replace(self.tmp_path, self.path)
        logger.info(f"State saved to {self.path}")

    def unserialize(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.state = torch.load(self.path, map_location=device, weights_only=False)
        logger.info(f"State loaded from {self.path}")

    def unpack_state(self):
        raise NotImplementedError("This method should be implemented in subclasses.")

    def pack_state(self):
        raise NotImplementedError("This method should be implemented in subclasses.")


class StateSaverSubmodlib(StateSaverBase):
    def __init__(self, *args, **kwargs):
        submodlib_method = kwargs.get('submodlib_method')
        super().__init__(*args, **kwargs)

        inter_path = self.path
        self.path = inter_path + f"_submodlib_{submodlib_method}.pt"
        self.tmp_path = inter_path + f"_submodlib_{submodlib_method}_tmp.pt"

    def pack_state(self, query_batch_index, partials_list, opts_done, opts,
                   corp_size=None):
        self.state['query_batch_index'] = query_batch_index
        self.state['partials_list'] = partials_list
        self.state['opts_done'] = opts_done
        self.state['opts'] = opts
        self.state['corp_size'] = corp_size

    def unpack_state(self):
        return self.state['query_batch_id'], self.state['partials_list'], self.state['opt_done'], \
                self.state['opts'], self.state['corp_size']

    def unserialize(self):
        # The items saved are numpy arrays, so map location should explicitly be 'cpu'
        self.state = torch.load(self.path, map_location='cpu', weights_only=False)
        logger.info(f"State loaded from {self.path}")


class StateSaverGreedy(StateSaverBase):
    def __init__(self, *args, **kwargs):
        greedy_bs = kwargs.get('greedy_bs')
        super().__init__(*args, **kwargs)

        inter_path = self.path
        self.path = inter_path + f"_greedy_{greedy_bs}.pt"
        self.tmp_path = inter_path + f"_greedy_{greedy_bs}_tmp.pt"

    def pack_state(self, budget_iter, optvec, opt_indices, opts_scores):
        self.state['budget_iter'] = budget_iter
        self.state['optvec'] = optvec
        self.state['opt_indices'] = opt_indices
        self.state['opts_scores'] = opts_scores

    def unpack_state(self):
        return (self.state['budget_iter'],
                self.state['optvec'],
                self.state['opt_indices'],
                self.state['opts_scores'])