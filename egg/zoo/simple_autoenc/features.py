# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pickle
import torch.utils.data as data
import torch.nn.parallel
import os
import torch
import numpy as np


class _OneHotIterator:
    """
    >>> it_1 = _OneHotIterator(n_features=128, n_batches_per_epoch=2, batch_size=64, seed=1)
    >>> it_2 = _OneHotIterator(n_features=128, n_batches_per_epoch=2, batch_size=64, seed=1)
    >>> list(it_1)[0][0].allclose(list(it_2)[0][0])
    True
    >>> it = _OneHotIterator(n_features=8, n_batches_per_epoch=1, batch_size=4)
    >>> data = list(it)
    >>> len(data)
    1
    >>> batch = data[0]
    >>> x, y = batch
    >>> x.size()
    torch.Size([4, 8])
    >>> x.sum(dim=1)
    tensor([1., 1., 1., 1.])
    """

    def __init__(self, n_dim,  n_objects,  n_batches_per_epoch, batch_size, seed=None):
        self.n_batches_per_epoch = n_batches_per_epoch
        # self.n_features = n_features
        self.batch_size = batch_size
        self.n_dim = n_dim
        self.n_objects =  n_objects
        n_features = n_dim * n_objects
        self.probs = np.ones(n_features) / n_features
        self.batches_generated = 0
        self.random_state = np.random.RandomState(seed)

    def __iter__(self):
        return self

    def __next__(self):
        if self.batches_generated >= self.n_batches_per_epoch:
            raise StopIteration()
        #
        # batch_data = self.random_state.multinomial(1, self.probs, size=self.batch_size)
        # self.batches_generated += 1
        # return torch.from_numpy(batch_data).float(), torch.zeros(1)

        # batch
        for i in range(self.batch_size):
            # one input
            dims = []

            for idx in range(self.n_dim):
                # pick properties from scale(set) of values
                dimension = np.random.choice(self.n_objects,
                                             self.n_objects, replace=False)

                # print(dimension)

                # put min value in 0, 2, 4 at the current dimension?
                # the index of min properties
                where_min = np.argmin(dimension)
                # min_idx = double the dim
                min_idx = idx * 2
                dimension[[where_min, min_idx]] = dimension[[min_idx, where_min]]  # swap values at the two positions

                # put max value in 1, 3, 5 at the current dimension? (target place?)

                where_max = np.argmax(dimension)
                max_idx = min_idx + 1
                dimension[[where_max, max_idx]] = dimension[[max_idx, where_max]]
                dims.append(dimension)
            if i == 0:
                # initializa batch_data
                batch_data = np.array(dims).flatten()
                # print("initial batch shape:  " + str(batch_data.shape))
            else:
                # add up batch_data
                # print("batch shape:  " + str(batch_data.shape))
                batch_data = np.vstack([batch_data, np.array(dims).flatten()])


        self.batches_generated += 1

        return torch.from_numpy(batch_data).float(), torch.zeros(1)


class OneHotLoader(torch.utils.data.DataLoader):
    """
    >>> data_loader = OneHotLoader(n_features=8, batches_per_epoch=3, batch_size=2, seed=1)
    >>> epoch_1 = []
    >>> for batch in data_loader:
    ...     epoch_1.append(batch)
    >>> [b[0].size() for b in epoch_1]
    [torch.Size([2, 8]), torch.Size([2, 8]), torch.Size([2, 8])]
    >>> data_loader_other = OneHotLoader(n_features=8, batches_per_epoch=3, batch_size=2)
    >>> all_equal = True
    >>> for a, b in zip(data_loader, data_loader_other):
    ...     all_equal = all_equal and (a[0] == b[0]).all()
    >>> all_equal.item()
    0
    """

    def __init__(self, n_dim , n_objects, batches_per_epoch, batch_size, seed=None):
        self.seed = seed
        self.batches_per_epoch = batches_per_epoch
        self.n_dim = n_dim
        self.n_objects = n_objects
        # self.n_features = n_dim * n_properties
        self.batch_size = batch_size

    def __iter__(self):
        if self.seed is None:
            seed = np.random.randint(0, 2 ** 32)
        else:
            seed = self.seed
        return _OneHotIterator(n_dim=self.n_dim, n_objects= self.n_objects, n_batches_per_epoch=self.batches_per_epoch,
                               batch_size=self.batch_size, seed=seed)
        # return _OneHotIterator(n_features=self.n_features, n_batches_per_epoch=self.batches_per_epoch,
        #                        batch_size=self.batch_size, seed=seed)
