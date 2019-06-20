from __future__ import absolute_import
from __future__ import division

from collections import defaultdict
import numpy as np
import copy
import random

import torch
from torch.utils.data.sampler import Sampler, RandomSampler


class RandomIdentitySampler(Sampler):
    """Randomly samples N identities each with K instances.

    Args:
        data_source (list): contains tuples of (img_path(s), pid, camid).
        batch_size (int): batch size.
        num_instances (int): number of instances per identity in a batch.
    """
    def __init__(self, data_source, batch_size, num_instances, seed=None):
        if batch_size < num_instances:
            raise ValueError('batch_size={} must be no less '
                             'than num_instances={}'.format(batch_size, num_instances))

        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        for index, (_, pid, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        self.seed = seed

        # estimate number of examples in an epoch
        # TODO: improve precision
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        """
        During validation, this sampler will reset the same random seed every epoch
        so that the validation sampler will give the same samples each epoch. However,
        during training, we do NOT want the same samples each epoch.
        Note that build_train_sampler does NOT set a seed, in which case self.seed
        will be None, whereas build_validation_sampler DOES set a seed, so
        self.seed will not be None.
        """

        if self.seed is not None: # i.e., during validation
            # Create temporary random instances for random and np.random
            # so that we don't alter the global random seeds
            rand = random.Random(self.seed)
            np_rand = np.random.RandomState(self.seed)

        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                if self.seed is not None:
                    idxs = np_rand.choice(idxs, size=self.num_instances, replace=True)
                else:
                    idxs = np.random.choice(idxs, size=self.num_instances, replace=True)

            if self.seed is not None:
                rand.shuffle(idxs)
            else:
                random.shuffle(idxs)

            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            if self.seed is not None:
                selected_pids = rand.sample(avai_pids, self.num_pids_per_batch)
            else:
                selected_pids = random.sample(avai_pids, self.num_pids_per_batch)

            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        return iter(final_idxs)

    def __len__(self):
        return self.length


def build_train_sampler(data_source, train_sampler, batch_size=32, num_instances=4):
    """Builds a training sampler.

    Args:
        data_source (list): contains tuples of (img_path(s), pid, camid).
        train_sampler (str): sampler name
        batch_size (int, optional): batch size. Default is 32.
        num_instances (int, optional): number of instances per identity in a
            batch (for ``RandomIdentitySampler``). Default is 4.
    """
    if train_sampler == 'RandomIdentitySampler':
        sampler = RandomIdentitySampler(data_source, batch_size, num_instances)
    
    elif train_sampler == "RandomSampler":
        sampler = RandomSampler(data_source)
    else:
        raise Exception("Please select a valid sampler")

    return sampler


def build_validation_sampler(data_source, sampler, seed, batch_size=32, num_instances=4):
    """Builds a validation sampler.

    Args:
        data_source (list): contains tuples of (img_path(s), pid, camid).
        sampler (str): sampler name
        batch_size (int, optional): batch size. Default is 32.
        num_instances (int, optional): number of instances per identity in a
            batch (for ``RandomIdentitySampler``). Default is 4.
        seed: random seed for the sampler (only used during validation, not training,
            to ensure that the validation samples for each epoch are identical)
    """
    if not sampler == "RandomIdentitySampler":
        raise Exception("Validation sampler must be RandomIdentitySampler because"
                        "only triplet loss is supported for validation")
    return RandomIdentitySampler(data_source, batch_size, num_instances, seed)