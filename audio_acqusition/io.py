import os.path
import h5py


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
        if os.path.exists(filename):
            # TODO: raise warning and continue
            idx = 1
            while os.path.exists('{}_{}{}'.format(name, idx, ext)):
                idx += 1
            name = '{}_{}'.format(name, idx)
            filename = name + ext

        self.filename = filename
        self.file = h5py.File(self.filename, mode='x')
        self.dataset = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def start_new_dataset(self, dataset=None, chunkshape=None):
        if dataset is None:
            dataset = 'data'  # Default name for sets
        if chunkshape is None:
            chunkshape = (4, 1024)  # Default chunkshape
        if dataset in self.file:
            idx = 1
            while '{}_{}'.format(dataset, idx) in self.file:
                idx += 1
            dataset = '{}_{}'.format(dataset, idx)

        self.dataset = self.file.create_dataset(dataset, shape=(0, 0), maxshape=(None, None), dtype='f8', chunks=chunkshape)

    def write(self, data):
        # TODO: make sure that data is a ndarray?
        shape = data.shape
        if len(shape) == 1: # Add new axis to the data
            data.shape = (1, -1)
            shape = data.shape
        if self.dataset is None:
            self.start_new_dataset()
            self.dataset.resize(shape[0], axis=0)
        currshape = self.dataset.shape
        newshape = (max(currshape[0], shape[0]), currshape[1] + shape[1])
        self.dataset.resize(newshape)
        self.dataset[:shape[0], -shape[1]:] = data

    def close(self):
        if self.file.id:
            self.file.close()


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
