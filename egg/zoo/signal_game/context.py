"""
Copyright (C) 2018 Shane Steinert-Threlkeld
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>
"""
import numpy as np


class Context(object):

    def __init__(self, n_dims, scale, n_objs=None, dims_per_obj=None,
                 shuffle=True):
        # TODO: allow non-trivial dims_per_obj and n_objs
        self.dims_per_obj = dims_per_obj or n_dims
        self.n_objs = n_objs or 2*n_dims
        self.n_dims = n_dims
        dims = []
        for idx in range(n_dims):
        	# pick properties from scale(set) of values
            dimension = np.random.choice(scale,
                                         size=self.n_objs, replace=False)
            
            print(dimension)
            
            # put min value in 0, 2, 4 at the current dimension?
            
            # the index of min properties
            where_min = np.argmin(dimension)
            # min_idx = double the dim
            min_idx = idx * 2
            dimension[[where_min, min_idx]] = dimension[[min_idx, where_min]] # swap values at the two positions
            
            # put max value in 1, 3, 5 at the current dimension? (target place?)
            
            where_max = np.argmax(dimension)
            max_idx = min_idx + 1
            dimension[[where_max, max_idx]] = dimension[[max_idx, where_max]]
            dims.append(dimension)

        self.dims = np.array(dims)

        if shuffle:
            objs = np.transpose(self.dims)
            np.random.shuffle(objs)
            self.dims = np.transpose(objs)

    def view(self, dims=None, dim_first=False, at_dim_idx=False):
        if dims is None:
            dims = self.dims
        if at_dim_idx:
            dims = (np.repeat(dims, self.n_dims, axis=1) *
                    np.tile(np.eye(self.n_dims), self.n_objs))
        if not dim_first:
            dims = np.transpose(dims)
        return dims.flatten()

    def dir_and_dim(self):
        as_min_max = np.stack([
            np.argmin(self.dims, axis=1),
            np.argmax(self.dims, axis=1)
        ])
        # a pair: first elt == 0 if min, 1 if max
        # second elt == dim
        direction, dim = np.where(as_min_max == 0)
        return direction[0], dim[0]

    def permuted_dims(self, perm):
        """Permutes self.dims by perm, where perm is a permutation of n_objs.
        Returns an np array of same shape as self.dims. """
        objs = np.transpose(self.dims)
        objs = objs[perm]
        return np.transpose(objs)
