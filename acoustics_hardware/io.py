import os.path
import h5py
import numpy as np


class HDFWriter:
    """ Implements writing to HDF 5 files """

    def __init__(self, filename=None):
        if filename is None:
            # TODO: Raise warning and continue
            filename = 'data'  # Default filename

        name, ext = os.path.splitext(filename)
        if len(ext) < 2:  # Either empty or just a dot
            ext = '.h5'
            filename = name + ext
        # if os.path.exists(filename):
        #     # TODO: raise warning and continue
        #     idx = 1
        #     while os.path.exists('{}_{}{}'.format(name, idx, ext)):
        #         idx += 1
        #     name = '{}_{}'.format(name, idx)
        #     filename = name + ext
        self.filename = filename
        self.file = h5py.File(self.filename, mode='a')
        self.group = self.file['/']
        self.dataset = None

        self.head = None
        self.data_shape = None

    def select_group(self, group=None):
        if group is None:
            group = '/'
        self.group = self.file.require_group(group)

    def create_dataset(self, name=None, ndim=2, **kwargs):
        if name is None:
            name = 'data'  # Default name for sets
        kwargs.setdefault('shape', ndim * (1,))
        kwargs.setdefault('maxshape', ndim * (None,))
        kwargs.setdefault('dtype', 'float64')
        self.dataset = self.group.create_dataset(name=name, **kwargs)
        self.head = np.array(ndim * (0,))
        self.data_shape = None

    # def select_dataset(self, dataset=None, **kwargs):
    #     # Selecting existing datasets this easily does not seem like a good idea
    #     # There is no really good way to make sure that writing continues where it should
    #     # To leave all existing data and axis lengths untouched we would need to append new
    #     # data along the first axis, but that will **always** extend the last axis, withot
    #     # any possibility to say something else first.
    #     # It is probably better if users who want to reopen existing datasets do so manually
    #     # and manually sets the head or extends using 'step'
    #     if dataset is None:
    #         dataset = 'data'  # Default name for sets
    #     # if chunkshape is None:
    #     #     chunkshape = (4, 1024)  # Default chunkshape
    #     # if dataset in self.group:
    #     #     idx = 1
    #     #     while '{}_{}'.format(dataset, idx) in self.group:
    #     #         idx += 1
    #     #     dataset = '{}_{}'.format(dataset, idx)
    #     if dataset in self.group:
    #         self.dataset = self.group[dataset]
    #         self.data_shape = None
    #     else:
    #         self.create_dataset(dataset, **kwargs)
    #     # self.dataset = self.group.require_dataset(dataset, exact=False, **kwargs)

    def write(self, data):
        # TODO: make sure that data is a ndarray?
        self.data_shape = data.shape
        # if self.data_shape is None:
        #     self.data_shape = data.shape
        # elif not self.data_shape[:-1] == data.shape[:-1]:
        #     raise ValueError('data dimentions does not match specified dimention')
        # if len(shape) == 1:  # Add new axis to the data
        #     data.shape = (1, -1)
        #     shape = data.shape
        if self.dataset is None:
            self.create_dataset(ndim=data.ndim, chunks=data.shape)

        for idx in range(data.ndim):
            ax = self.head.size - data.ndim + idx
            if self.head[ax] + data.shape[idx] > self.dataset.shape[ax]:
                self.dataset.resize(self.head[ax] + data.shape[idx], axis=ax)

        # if any(self.head[-data.ndim:] + data.shape > self.dataset.shape[-data.ndim:]):
        #     # We need to resize
        #     # TODO: This could possibly be optimized to rezise less frequent for the last axis
        #     new_shape = np.array(self.dataset.shape)
        #     new_shape[-data.ndim:] = self.head[-data.ndim:] + data.shape
        #     # self.dataset.resize(new_length, axis=self.dataset.ndim - 1)
        #     self.dataset.resize(new_shape)

        # All indices exept the last ndim number are constant
        idx_list = list(self.head[:-data.ndim])
        # The last couple indices should be sliced from head to head+data.shape
        idx_list.extend([slice(start, start + length) for start, length in zip(self.head[-data.ndim:], data.shape)])
        # The list must be converted to a tuple
        self.dataset[tuple(idx_list)] = data
        # Uptade the head
        # self.head = np.array(idx_list[:-data.ndim].extend([s.stop for s in idx_list[-data.ndim:]]))
        self.head[-1] += data.shape[-1]

    def step(self, axis=-1):
        # if axis < 0:
        #    axis = self.dataset.ndim + axis
        if axis >= 0:
            # We would like to always index from the rear since the dimentions align there
            axis = axis - self.dataset.ndim

        # self.head = np.array(self.head)
        # data_shape = self.data_shape
        # self.data_shape = None

        if -axis <= len(self.data_shape):
            # Step along axis in data
            # Reshaping as a consequence of this cannot be done here since we allow a new data shape
            self.head[axis] += self.data_shape[axis]
        else:
            # Step along axis not existing in data, resize if we need to
            self.head[axis] += 1
            if self.head[axis] >= self.dataset.shape[axis]:
                self.dataset.resize(self.head[axis] + 1, self.dataset.ndim + axis)
        # Reset axes after step axis
        # Strange indexing needed for when axis=-1, since there is no -0 equivalent
        self.head[self.head.size + axis + 1:] = 0
        # self.head = head

    def _old_step(self, order=0):
        '''
        Moves the head to start a new measurement section
        'order' sets the number of additional dimentions that will be leaved by the step (default 0).

        Example for 2d data
        order=0, the next chunk of data will be placed at [..., +shape[1], 0] relative to previous
        order=1, the next chunk of data will be placed at [...,+1, 0, 0] relative to previous
        Example for 3d data
        order=0,  the next chunk of data will be placed at [..., +shape[2], 0, 0] relative to previous
        order=1,  the next chunk of data will be placed at [..., +1, 0, 0, 0] relative to previous

        In other words, ndim(dataset) needs to be at least ndim(data)+order.
        order>0 will preserve the length of the last ndim(data) axes.
        '''
        head = np.array(self.head)
        data_shape = self.data_shape
        if order == 0:
            ndim = len(data_shape)
            head[-ndim] += data_shape[0]
        else:
            ndim = len(data_shape) + order
            head[-ndim] += 1
        head[-(ndim - 1):] = 0
        self.head = tuple(head)

    def close(self):
        if self.file.id:
            self.file.close()

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()


class HDFReader:
    """ Implements reading from HDF 5 files"""

    def __init__(self, filename):
        if not os.path.isfile(filename):
            raise IOError('No such file: {}'.format(filename))
        self.file = h5py.File(filename, mode='r')
        self.datasets = []
        for dataset in self.file:
            self.datasets.append(dataset)
        self.dataset = self.datasets[0]

    def __del__(self):
        self.close()

    def close(self):
        if self.file.id:
            self.file.close()

    @property
    def shape(self):
        return self.dataset.shape

    @property
    def dataset(self):
        return self._dataset

    @dataset.setter
    def dataset(self, dataset):
        if dataset not in self.datasets:
            if isinstance(dataset, int):
                dataset = self.datasets[dataset]
            else:
                raise KeyError(dataset)
        self._dataset = self.file[dataset]

    def blocks(self, start=None, stop=None, blocksize=None):
        """ A block generator. The last block might be of a different shape than the rest."""
        if start is None:
            start = 0
        if stop is None:
            stop = self.shape[1]
        if start < 0:
            start = self.shape[1] + start
        if stop < 0:
            stop = self.shape[1] + stop
        if blocksize is None:
            blocksize = self.dataset.chunks[1]
        n_blocks = -((start - stop) // blocksize)  # Round up
        for block in range(n_blocks):
            yield self[:, start + block * blocksize:min(start + (block + 1) * blocksize, stop)]

    def __getitem__(self, key):
        if len(key) > self.dataset.ndim:
            self.dataset = (key[-1])
            return self.dataset[key[:-1]]
        else:
            return self.dataset[key]
