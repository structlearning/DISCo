import os
import pickle


class StateSaverBase(object):
    def __init__(self, *args, **kwargs):
        self.dataset = kwargs.get('dataset')
        super().__init__()
        self.state = {}

        self.prefix = "results/intermediate"
        os.makedirs(self.prefix, exist_ok=True)

        self.path = f"{self.prefix}/{self.dataset}"

    def serialize(self):
        with open(self.path, 'wb') as f:
            pickle.dump(self.state, f)

    def unserialize(self):
        with open(self.path, 'rb') as f:
            self.state = pickle.load(f)

    def unpack_state(self):
        raise NotImplementedError("This method should be implemented in subclasses.")

    def pack_state(self):
        raise NotImplementedError("This method should be implemented in subclasses.")


class StateSaverSubmodlib(StateSaverBase):
    def __init__(self, *args, **kwargs):
        submodlib_method = kwargs.get('submodlib_method')
        super().__init__(*args, **kwargs)

        self.path = self.path + f"_submodlib_{submodlib_method}.pt"

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


class StateSaverGreedy(StateSaverBase):
    def __init__(self, *args, **kwargs):
        greedy_bs = kwargs.get('greedy_bs')
        super().__init__(*args, **kwargs)

        self.path = self.path + f"_greedy_{greedy_bs}.pt"

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
