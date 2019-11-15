import queue
import numpy as np
import sounddevice as sd
from . import core

import logging
logger = logging.getLogger(__name__)

try:
    import nidaqmx
    import nidaqmx.stream_readers
    import nidaqmx.stream_writers
    import warnings
    warnings.filterwarnings('ignore', category=nidaqmx.errors.DaqWarning)
except ImportError:
    pass


class AudioDevice(core.Device):
    """Class for interacting with audio interfaces.

    Implementation of the `~.core.Device` framework for audio interfaces.
    Built on top of the `sounddevice <http://python-sounddevice.readthedocs.io/>`_ package.

    Arguments:
        name (`str`): Partial or full name of the audio interface.
        fs (`float`, optional): Sample rate for the device, defaults to system default for the device.
        framesize (`int`, optional): The framesize for inputs and outputs, defaults to 1024 samples.
    """
    @staticmethod
    def get_devices(name=None):
        """Check which audio interfaces can be interacted with

        Arguments:
            name (`str`, optional): incomplete name of device or ``None``
        Returns:
            Complete name of device, or list of all devices.
        Todo:
            - The framesize by default set to 1024, is this appropriate?
            - `sounddevice` allows for variable framesizes, can we make use of this somehow?
        """
        if name is None:
            return sd.query_devices()
        else:
            return sd.query_devices(name)['name']

    def __init__(self, name=None, fs=None, framesize=None, **kwargs):
        super().__init__(**kwargs)
        if name is None:
            self.name = AudioDevice.get_devices()[0]['name']
        else:
            self.name = AudioDevice.get_devices(name)

        if fs is None:
            self.fs = sd.query_devices(self.name)['default_samplerate']
        else:
            self.fs = fs

        if framesize is None:
            self.framesize = 1024
        else:
            self.framesize = framesize

        self.inputs = []
        self.outputs = []

    def _hardware_run(self):
        while sd._initialized > 0:
            sd._terminate()
        sd._initialize()
        # Create streams
        num_output_ch = max(self.outputs, default=-1) + 1
        num_input_ch = max(self.inputs, default=-1) + 1
        self.silent_ch = [ch for ch in range(num_output_ch) if ch not in self.outputs]
        self._hardware_output_timeout = 0.5 * self.framesize / self.fs
        if num_input_ch and num_output_ch:
            # in/out stream
            logger.debug('Creating input/output hardare')
            def callback(indata, outdata, frames, time, status):
                self._input_callback(indata)
                self._output_callback(outdata)
            stream = sd.Stream(device=self.name, samplerate=self.fs, blocksize=self.framesize, callback=callback, channels=(num_input_ch, num_output_ch))
        elif num_input_ch:
            logger.debug('Creating input hardare')
            def callback(indata, frames, time, status):
                self._input_callback(indata)
            stream = sd.InputStream(device=self.name, samplerate=self.fs, blocksize=self.framesize, callback=callback, channels=num_input_ch)
        elif len(self.outputs):
            logger.debug('Creating output hardare')
            def callback(outdata, frames, time, status):
                self._output_callback(outdata)
            stream = sd.OutputStream(device=self.name, samplerate=self.fs, blocksize=self.framesize, callback=callback, channels=num_output_ch)
        else:
            # Do something??
            pass

        stream.start()
        logger.verbose('Hardare running')
        self._sync_event.set()
        self._hardware_stop_event.wait()
        stream.stop()

    def _input_callback(self, indata):
        self._hardware_input_Q.put(indata.T[self.inputs].copy())  # Transpose before copy to actually reverse the memory layout

    def _output_callback(self, outdata):
        try:
            outdata[:, self.outputs] = self._hardware_output_Q.get(timeout=self._hardware_output_timeout).T
            outdata[:, self.silent_ch] = 0
        except queue.Empty:
            outdata[:] = 0
        else:
            self._hardware_output_Q.task_done()

    @property
    def max_inputs(self):
        return sd.query_devices(self.name)['max_input_channels']

    @property
    def max_outputs(self):
        return sd.query_devices(self.name)['max_output_channels']


class NIDevice(core.Device):
    """Class for interacting with national instruments hardware.

    Implementation of the `~.core.Device` framework for national instruments hardware.
    Built on top of the `nidaqmx <https://nidaqmx-python.readthedocs.io/>`_ package.

    Arguments:
        name (`str`): Partial or full name of the audio interface.
        fs (`float`, optional): Sample rate for the device, defaults maximum supported rate.
        framesize (`int`, optional): The framesize for inputs and outputs, defaults to 10000 samples.
        dtype (`str`, optional): The datatype used while reading input data, default ``'float64'``.
    Todo:
        The framesize needs reasonable defaults, and needs to be protected
        by validity checks.
    """
    @classmethod
    def get_devices(cls, name=None):
        """Check which NI hardware is available.

        Since `NIDevice` is intended to interact with a single module at the
        time, this will complete names by selecting the first available module
        is a chassi.

        Arguments:
            name (`str`, optional): incomplete name of device or ``None``
        Returns:
            Complete name of device, or list of all devices.
        """
        try:
            system = nidaqmx.system.System.local()
        except NameError as e:
            if e.args[0] == "name 'nidaqmx' is not defined":
                raise ModuleNotFoundError("Windows-only module 'nidaqmx' is not installed")
            else:
                raise e
        name_list = [dev.name for dev in system.devices]
        if name is None:
            return name_list
        else:
            if len(name) == 0:
                name = name_list[0]
            return name

    @classmethod
    def get_chassis(cls, name=None):
        devs = cls.get_devices()
        chassis = [dev for dev in devs if 'cDAQ' in dev and 'Mod' not in dev]
        if name is None:
            return chassis
        if name in chassis:
            return name
        return [dev for dev in chassis if str(name) in dev][0]

    @classmethod
    def get_modules(cls, chassis):
        chassis = cls.get_chassis(chassis)
        devs = cls.get_devices()
        return [dev for dev in devs if chassis in dev and 'Mod' in dev]

    @classmethod
    def reset_chassis(cls, chassis):
        nidaqmx.system.Device(chassis).reset_device()

    @classmethod
    def reset_chassis_modules(cls, chassis):
        for module in cls.get_modules(chassis):
            nidaqmx.system.Device(module).reset_device()

    def __init__(self, name=None, fs=None, framesize=10000, dtype='float64', **kwargs):
        super().__init__(**kwargs)
        if name is None:
            name = NIDevice.get_devices()[0]
        self.name = name
        self.chassis = self.get_chassis(name)
        self.modules = self.get_modules(self.chassis)
        self.input_modules = [module for module in self.modules if len(nidaqmx.system.Device(module).ai_physical_chans) > 0]
        self.output_modules = [module for module in self.modules if len(nidaqmx.system.Device(module).ao_physical_chans) > 0]
        if fs is None:
            if len(self.input_modules) > 0:
                self.fs = nidaqmx.system.Device(self.input_modules[0]).ai_max_single_chan_rate
            elif len(self.output_modules) > 0:
                self.fs = nidaqmx.system.Device(self.output_modules[0]).ao_max_rate
            else:
                raise AttributeError('Device has no modules!')
        else:
            self.fs = fs
        self.framesize = framesize
        self.dtype = dtype

    @property
    def max_inputs(self):
        inputs = 0
        for module in self.input_modules:
            inputs += len(nidaqmx.system.Device(module).ai_physical_chans)
        return inputs

    @property
    def max_outputs(self):
        outputs = 0
        for module in self.output_modules:
            outputs += len(nidaqmx.system.Device(module).ao_physical_chans)
        return outputs

    def input_map(self, index=None, module=None, channel=None):
        if index is not None:
            # Mapping from index to module and channel
            module = 0
            channel = index
            while channel >= len(nidaqmx.system.Device(self.input_modules[module]).ai_physical_chans):
                channel -= len(nidaqmx.system.Device(self.input_modules[module]).ai_physical_chans)
                module += 1
            return module, channel
        elif channel is not None and module is not None:
            # Mapping from channel and module to index
            index = 0
            for mod in range(module-1):
                index += len(nidaqmx.system.Device(self.input_modules[mod]).ai_physical_chans)
            index += channel
            return index
        else:
            mapping = []
            for input in self.inputs:
                module, channel = self.input_map(index=input)
                mapping.append(nidaqmx.system.Device(self.input_modules[module]).ai_physical_chans[channel].name)
            return mapping

    def output_map(self, index=None, module=None, channel=None):
        if index is not None:
            # Mapping from index to module and channel
            module = 0
            channel = index
            while channel >= len(nidaqmx.system.Device(self.output_modules[module]).ao_physical_chans):
                channel -= len(nidaqmx.system.Device(self.output_modules[module]).ao_physical_chans)
                module += 1
            return module, channel
        elif channel is not None and module is not None:
            # Mapping from channel and module to index
            index = 0
            for mod in range(module-1):
                index += len(nidaqmx.system.Device(self.output_modules[mod]).ao_physical_chans)
            index += channel
            return index
        else:
            mapping = []
            for output in self.outputs:
                module, channel = self.output_map(index=output)
                mapping.append(nidaqmx.system.Device(self.output_modules[module]).ao_physical_chans[channel].name)
            return mapping

    def bit_depth(self, channel=None):
        """The bitdepth for the device.

        Currently only implemented for input devices.

        Todo:
            - What would be the expected behavior if the channels have different depths?
            - Chack that the task exists, else warn.
        """
        if channel is None:
            ch_idx = 0
        else:
            ch_idx = self.inputs.index(channel)
        return self._input_task.ai_channels[ch_idx].ai_resolution

    def word_length(self, channel=None):
        """The word length for the device.

        Only valid when using raw, unscaled, data types.
        Currently only implemented for input devices.

        Todo:
            - Chack that the task exists, else warn.
        """
        if channel is None:
            return max([ch.ai_raw_samp_size for ch in self._input_task.ai_channels])
        else:
            ch_idx = self.inputs.index(channel)
            return self._input_task.ai_channels[ch_idx].ai_raw_samp_size

    def scaling_coeffs(self, channels=None):
        """Scaling coefficients used while reading raw input.

        Returns the polynomial coefficients required to calculate
        input voltage from unscaled integers.

        Todo:
            - Chack that the task exists, else warn.
        """
        if channels is None:
            channels = self.inputs
        try:
            ch_idx = [self.inputs.index(ch) for ch in channels]
        except TypeError:
            ch_idx = self.inputs.index(channels)
            return self._input_task.ai_channels[ch_idx].ai_dev_scaling_coeff
        else:
            return [self._input_task.ai_channels[idx].ai_dev_scaling_coeff for idx in ch_idx]

    def _hardware_run(self):
        self.reset_chassis_modules(self.chassis)
        self._input_task = nidaqmx.Task()
        self._output_task = nidaqmx.Task()
        self.__hardware_input_frames = 0
        self.__hardware_output_frames = 0

        for ch in self.inputs:
            module, channel = self.input_map(index=ch)
            self._input_task.ai_channels.add_ai_voltage_chan(self.input_modules[module] + '/ai{}'.format(int(channel)))
        for ch in self.outputs:
            module, channel = self.output_map(index=ch)
            output_range = nidaqmx.system.Device(self.output_modules[module]).ao_voltage_rngs
            self._output_task.ao_channels.add_ao_voltage_chan(self.output_modules[module] + '/ao{}'.format(int(channel)), min_val=output_range[0], max_val=output_range[1])
        if len(self.inputs) and len(self.outputs):
            self._input_task.timing.cfg_samp_clk_timing(int(self.fs), samps_per_chan=self.framesize,
                    sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS)
            self._output_task.timing.cfg_samp_clk_timing(int(self.fs), samps_per_chan=self.framesize,
                    sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS)
            self._output_task.timing.samp_clk_timebase_src = self._input_task.timing.samp_clk_timebase_src
            self._output_task.timing.samp_clk_timebase_rate = self._input_task.timing.samp_clk_timebase_rate
        elif len(self.inputs):
            self._input_task.timing.cfg_samp_clk_timing(int(self.fs), samps_per_chan=self.framesize,
                    sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS)
        elif len(self.outputs):
            self._output_task.timing.cfg_samp_clk_timing(int(self.fs), samps_per_chan=self.framesize,
                    sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS)

        if self.dtype.lower() == 'unscaled' or self.dtype.lower() == 'int':
            wl = self.word_length()
            self.dtype = 'int{}'.format(int(wl))

        if len(self.inputs):
            if self.dtype == 'int16':
                reader = nidaqmx.stream_readers.AnalogUnscaledReader(self._input_task.in_stream)
                databuffer = np.empty((len(self.inputs), self.framesize), dtype='int16')
                read_function = reader.read_int16
            elif self.dtype == 'int32':
                reader = nidaqmx.stream_readers.AnalogUnscaledReader(self._input_task.in_stream)
                databuffer = np.empty((len(self.inputs), self.framesize), dtype='int32')
                read_function = reader.read_int32
            elif self.dtype == 'uint16':
                reader = nidaqmx.stream_readers.AnalogUnscaledReader(self._input_task.in_stream)
                databuffer = np.empty((len(self.inputs), self.framesize), dtype='uint16')
                read_function = reader.read_uint16
            elif self.dtype == 'uint32':
                reader = nidaqmx.stream_readers.AnalogUnscaledReader(self._input_task.in_stream)
                databuffer = np.empty((len(self.inputs), self.framesize), dtype='uint32')
                read_function = reader.read_uint32
            else:  # Read as scaled float64
                reader = nidaqmx.stream_readers.AnalogMultiChannelReader(self._input_task.in_stream)
                databuffer = np.empty((len(self.inputs), self.framesize), dtype='float64')
                read_function = reader.read_many_sample

            def input_callback(task_handle, every_n_samples_event_type,
                               number_of_samples, callback_data):
                sampsRead = read_function(databuffer, self.framesize)
                self._hardware_input_Q.put(databuffer.copy())
                self.__hardware_input_frames += 1
                return 0
            self._input_task.register_every_n_samples_acquired_into_buffer_event(self.framesize, input_callback)
            self._input_task.start()
            logger.debug('Hardware input initialized')

        if len(self.outputs):
            writer = nidaqmx.stream_writers.AnalogMultiChannelWriter(self._output_task.out_stream)
            write_funciton = writer.write_many_sample
            self._output_task.out_stream.regen_mode = nidaqmx.constants.RegenerationMode.DONT_ALLOW_REGENERATION  # Needed to prevent issues with buffer overwrites and reuse
            write_funciton(np.zeros((self._output_task.out_stream.num_chans, 2*self.framesize)))  # Pre-fill the buffer with zeros, there needs to be something in the buffer when we start
            timeout = 0.5 * self.framesize / self.fs
            zero_frame = np.zeros((self._output_task.out_stream.num_chans, self.framesize))

            def output_callback(task_handle, every_n_samples_event_type,
                                number_of_samples, callback_data):
                try:
                    data = self._hardware_output_Q.get(timeout=timeout)
                except queue.Empty:
                    data = zero_frame
                    logger.frames('Hardware output Q empty')
                else:
                    self._hardware_output_Q.task_done()
                finally:
                    sampsWritten = write_funciton(data)
                    self.__hardware_output_frames += 1
                return 0
            self._output_task.register_every_n_samples_transferred_from_buffer_event(self.framesize, output_callback)
            self._output_task.start()
            logger.debug('Hardware output initialized')

        logger.verbose('Hardware running')
        self._sync_event.set()
        self._hardware_stop_event.wait()

        logger.debug('Hardware terminating')
        self._input_task.stop()
        self._output_task.stop()
        logger.debug('Hardware tasks stopped')
        self._input_task.wait_until_done(timeout=10)
        self._output_task.wait_until_done(timeout=10)
        logger.debug('Hardware tasks done')
        self._input_task.close()
        self._output_task.close()
        logger.debug('Hardware tasks closed')
        self.reset_chassis_modules(self.chassis)
        logger.verbose('Hardware terminated')


class FeedbackDevice(core.Device):
    def _hardware_run(self):
        while not self._hardware_stop_event.is_set():
            try:
                self._hardware_input_Q.put(self._hardware_output_Q.get(timeout=self._hardware_timeout))
            except queue.Empty:
                pass
