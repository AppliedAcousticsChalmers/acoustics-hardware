import os.path
import h5py
import numpy as np
import multiprocessing
import threading
import queue


class HDFWriter:
    """ Implements writing to HDF 5 files """
    _timeout = 0.1

    def __init__(self, filename=None):
        if filename is None:
            filename = 'data'  # Default filename

        name, ext = os.path.splitext(filename)
        if len(ext) < 2:  # Either empty or just a dot
            ext = '.h5'
            filename = name + ext
        self.filename = filename
        self._file = None
        self._group = None
        self._devices = []
        self._input_Qs = []
        self._internal_Q = multiprocessing.Queue()
        self._datasets = []
        self._manual_dataset = None

        self._stop_event = threading.Event()

    def add_input(self, device=None, Q=None):
        if device is None and Q is None:
            raise ValueError('Either `device` or `Q` must be given as input')
        if Q is None:
            Q = device.input_Q
        self._devices.append(device)
        self._input_Qs.append(Q)

    def start(self, mode='auto', use_process=True):
        self.mode = mode
        if use_process:
            self._process = multiprocessing.Process(target=self._write_target)
        else:
            self._process = threading.Thread(target=self._write_target)
        self._process.start()
        if mode == 'auto':
            self._thread = threading.Thread(target=self._auto_target)
            self._thread.start()
        elif mode == 'signal':
            self._write_signal = threading.Event()
            self._thread = threading.Thread(target=self._signal_target)
            self._thread.start()
        elif mode == 'manual':
            class Dummy_thread:
                def join():
                    return True
            self._thread = Dummy_thread

    def stop(self):
        self._stop_event.set()
        self._thread.join()
        self.write_device_configs()
        self._internal_Q.put(None)
        self._stop_event.clear()

    def select_group(self, **kwargs):
        self._internal_Q.put(('select', kwargs))

    def _select_group(self, group=None):
        if group is None:
            group = '/'
        self._group = self._file.require_group(group)

    def create_dataset(self, **kwargs):
        self._internal_Q.put(('create', kwargs))

    def _create_dataset(self, name=None, ndim=2, index=None, **kwargs):
        if name is None:
            # TODO: What if the set already exists?
            name = 'data{}'.format(index).replace('None', '')  # Default name for sets
        if name in self._group:
            name_idx = 0
            while name + '_' + str(name_idx) in self._group:
                name_idx += 1
            name = name + '_' + str(name_idx)
        kwargs.setdefault('shape', ndim * (1,))
        kwargs.setdefault('maxshape', ndim * (None,))
        kwargs.setdefault('dtype', 'float64')
        dataset = [self._group.create_dataset(name=name, **kwargs), np.array(ndim * (0,)), None]
        if index is None:
            self._manual_dataset = dataset
        elif len(self._datasets) > index:
            self._datasets[index] = dataset
        else:
            skips = index - len(self._datasets)
            self._datasets.extend(skips * [None])
            self._datasets.append(dataset)
        return dataset

    def write(self, **kwargs):
        # TODO: debug the step keyword
        # We want it to work both for True/False in both manual writes and signal mode,
        # as well as with an index / a list of indices for the signalled mode
        if self.mode == 'auto':
            # TODO: raise?
            pass
        elif self.mode == 'signal':
            if 'step' in kwargs:
                for idx in range(len(self._input_Qs)):
                    self.step(axis=kwargs['step'], index=idx)
            self._write_signal.set()
        elif self.mode == 'manual':
            # TODO: Enable manual writes regardless of mode
            # If the write function is called with some data or a Q,
            # write that data to the manual dataset
            idx = kwargs.pop('index', None)
            if 'name' in kwargs:
                create_args = kwargs.copy()
                [create_args.pop(key, None) for key in ['Q', 'data', 'step']]
                self.create_dataset(create_args)
            if 'Q' in kwargs:
                while True:
                    try:
                        self._internal_Q.put(('write', (kwargs['Q'].get(timeout=self._timeout), idx)))
                    except queue.Empty:
                        break
            elif 'data' in kwargs:
                self._internal_Q.put(('write', (kwargs['data'], idx)))
            if 'step' in kwargs:
                self.step(axis=kwargs['step'])

    def _write_attrs(self, index=None, **kwargs):
        if index is None:
            attrs = self._group.attrs
        else:
            attrs = self._datasets[index].attrs
        for key, value in kwargs.items():
            attrs[key] = value

    def write_attrs(self, **kwargs):
        self._internal_Q.put(('attrs', kwargs))

    def write_device_configs(self, index=None, *args, **kwargs):
        if index is None:
            for idx in range(len(self._devices)):
                self.write_device_configs(index=idx, *args, **kwargs)
        else:
            attr_names = {'fs', 'label', 'name', 'serial_number'}
            device = self._devices[index]
            # Qs without devices will correspond to None here, so attrs will be an empty dict => only kwargs will be written
            attrs = {}
            for attr in attr_names.union(args):
                try:
                    value = getattr(device, attr)
                except AttributeError:
                    pass
                else:
                    attrs[attr] = value
            try:
                inputs = device.inputs
            except AttributeError:
                pass
            else:
                attrs['input_channels'] = np.string_([ch.to_json() for ch in inputs])
            try:
                outputs = device.outputs
            except AttributeError:
                pass
            else:
                attrs['output_channels'] = np.string_([ch.to_json() for ch in outputs])
            attrs.update(kwargs)
            self.write_attrs(index=index, **attrs)

    def _write(self, data, index=None):
        # TODO: make sure that data is a ndarray?
        if index is None:
            # Not an automated write from device Q
            try:
                dataset, head, _ = self._manual_dataset
            except TypeError:
                # We get `TypeError` if `manual_datast` is `None`
                dataset, head, _ = self._create_dataset(ndim=data.ndim, chunks=data.shape)
            self._manual_dataset[2] = data.shape
        else:
            try:
                dataset, head, _ = self._datasets[index]
            except (IndexError, TypeError):
                # We will get an IndexError if index indicated a hogher number dataset than what already exists,
                # but if we have a sparse creation of sets, e.g. set 0 is missing, but set 1 exists, we will get
                # a type error since `None` is not iterable
                dataset, head, _ = self._create_dataset(ndim=data.ndim, chunks=data.shape, index=index)
            self._datasets[index][2] = data.shape

        for idx in range(data.ndim):
            ax = head.size - data.ndim + idx
            if head[ax] + data.shape[idx] > dataset.shape[ax]:
                dataset.resize(head[ax] + data.shape[idx], axis=ax)

        # All indices exept the last ndim number are constant
        idx_list = list(head[:-data.ndim])
        # The last indices should be sliced from head to head+data.shape
        idx_list.extend([slice(start, start + length) for start, length in zip(head[-data.ndim:], data.shape)])
        # The list must be converted to a tuple
        dataset[tuple(idx_list)] = data
        # Uptade the head
        head[-1] += data.shape[-1]

    def step(self, **kwargs):
        self._internal_Q.put(('step', kwargs))

    def _step(self, axis=True, index=None):
        if index is None:
            try:
                dataset, head, data_shape = self._manual_dataset
            except TypeError:
                # TODO: This happens if someone tries to step before datasets are created
                return
        else:
            try:
                dataset, head, data_shape = self._datasets[index]
            except TypeError:
                # TODO: TypeError if someone tries to step before datasets are created
                return

        if isinstance(axis, bool):
            axis = max(len(head) - len(data_shape) - 1, 0)
        if axis < 0:
            axis = len(head) + axis - 1
        steps = (len(head) - len(data_shape)) * (1,) + data_shape
        head[axis] += steps[axis]
        head[axis + 1:] = 0
        if axis < len(head) - len(data_shape) and head[axis] >= dataset.shape[axis]:
            dataset.resize(head[axis] + 1, axis)

        return
        # Old implementation below
        if isinstance(axis, bool):
            # Leave the data dimentions intact, step along the next one
            axis = -len(data_shape) - 1
        if axis >= 0:
            # We would like to always index from the rear since the dimentions align there
            axis = axis - dataset.ndim

        if -axis <= len(data_shape):
            # Step along axis in data
            # Reshaping as a consequence of this cannot be done here since we allow a new data shape
            head[axis] += data_shape[axis]
        else:
            # Step along axis not existing in data, resize if we need to
            head[axis] += 1
            if head[axis] >= dataset.shape[axis]:
                dataset.resize(head[axis] + 1, dataset.ndim + axis)
        # Reset axes after step axis
        # Strange indexing needed for when axis=-1, since there is no -0 equivalent
        head[head.size + axis + 1:] = 0
        # self.head = head

    def _write_target(self):
        def handle(item):
            # TODO: Well, we need to handle the incomming items
            if item is None:
                raise StopIteration
            else:
                action = item[0]
                args = item[1]
            if action == 'write':
                self._write(*args)
            elif action == 'create':
                dataset = self._create_dataset(**args)
            elif action == 'step':
                self._step(**args)
            elif action == 'select':
                self._select_group(**args)

        self._file = h5py.File(self.filename, mode='a')
        self._select_group()
        # Main write loop
        while True:
            try:
                handle(self._internal_Q.get(timeout=self._timeout))
            except queue.Empty:
                continue
            except StopIteration:
                break
        self._file.close()

    def _auto_target(self):
        # TODO: Create named 2D datasets for the Qs/devices
        for idx, device in enumerate(self._devices):
            name = getattr(device, 'label', getattr(device, 'name', 'dataset{}'.format(idx)))
            self.create_dataset(name=name, ndim=2, index=idx)
        while not self._stop_event.is_set():
            for idx, Q in enumerate(self._input_Qs):
                try:
                    data = Q.get(timeout=self._timeout)
                except queue.Empty:
                    continue
                self._internal_Q.put(('write', (data, idx)))
        # Stop event have bel set, clear out any remaining stuff in th Q
        for idx, Q in enumerate(self._input_Qs):
            while True:
                try:
                    data = Q.get(timeout=self._timeout)
                except queue.Empty:
                    break
                self._internal_Q.put(('write', (data, idx)))

    def _signal_target(self):
        # TODO: Create named 3D datasets for all Qs
        for idx, device in enumerate(self._devices):
            name = getattr(device, 'label', getattr(device, 'name', 'dataset{}'.format(idx)))
            self.create_dataset(name=name, ndim=3, index=idx)
        # TODO: Enable ndim configuration
        while not self._stop_event.is_set():
            if self._write_signal.wait(self._timeout):
                self._write_signal.clear()
                for Q in self._input_Qs:
                    # We should acuire all the locks as fast a possible!
                    Q.mutex.acquire()
                for idx, Q in enumerate(self._input_Qs):
                    for data in Q.queue:
                        self._internal_Q.put(('write', (data, idx)))
                    Q.queue.clear()
                for Q in self._input_Qs:
                    Q.mutex.release()


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
