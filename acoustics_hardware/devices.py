import queue
import numpy as np
import sounddevice as sd
from . import core

try:
    import nidaqmx
    import nidaqmx.stream_readers
    import nidaqmx.stream_writers
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
            def callback(indata, outdata, frames, time, status):
                self._input_callback(indata)
                self._output_callback(outdata)
            stream = sd.Stream(device=self.name, samplerate=self.fs, blocksize=self.framesize, callback=callback, channels=(num_input_ch, num_output_ch))
        elif num_input_ch:
            def callback(indata, frames, time, status):
                self._input_callback(indata)
            stream = sd.InputStream(device=self.name, samplerate=self.fs, blocksize=self.framesize, callback=callback, channels=num_input_ch)
        elif len(self.outputs):
            def callback(outdata, frames, time, status):
                self._output_callback(outdata)
            stream = sd.OutputStream(device=self.name, samplerate=self.fs, blocksize=self.framesize, callback=callback, channels=num_output_ch)
        else:
            # Do something??
            pass

        stream.start()
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
    @staticmethod
    def get_devices(name=None):
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
            if name[:4] == 'cDAQ' and name[5:8] != 'Mod':
                name = [x for x in name_list if x[:8] == name[:5] + 'Mod'][0]
            return name

    def __init__(self, name=None, fs=None, framesize=10000, dtype='float64', **kwargs):
        super().__init__(**kwargs)
        if name is None:
            name = NIDevice.get_devices()[0]
        self.name = NIDevice.get_devices(name)
        if fs is None:
            try:
                self.fs = nidaqmx.system.Device(self.name).ai_max_single_chan_rate
            except nidaqmx.DaqError as e:
                if e.error_type == nidaqmx.error_codes.DAQmxErrors.ATTR_NOT_SUPPORTED:
                    self.fs = nidaqmx.system.Device(self.name).ao_max_rate
                else:
                    raise
        else:
            self.fs = fs
        self.framesize = framesize
        self.dtype = dtype

    @property
    def max_inputs(self):
        return len(nidaqmx.system.Device(self.name).ai_physical_chans)

    @property
    def max_outputs(self):
        return len(nidaqmx.system.Device(self.name).ao_physical_chans)

    @property
    def input_range(self):
        """Returns the input range on the device.

        Note:
            This is only an approximate value, do NOT use for calibrating
            unscaled readings.
        """
        return nidaqmx.system.Device(self.name).ai_voltage_rngs

    @property
    def output_range(self):
        """Returns the output range on the device.

        Note:
            This is only an approximate value, do NOT use for calibrating
            unscaled outputs.
        """
        return nidaqmx.system.Device(self.name).ao_voltage_rngs

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
        return self._task.ai_channels[ch_idx].ai_resolution

    def word_length(self, channel=None):
        """The word length for the device.

        Only valid when using raw, unscaled, data types.
        Currently only implemented for input devices.

        Todo:
            - Chack that the task exists, else warn.
        """
        if channel is None:
            return max([ch.ai_raw_samp_size for ch in self._task.ai_channels])
        else:
            ch_idx = self.inputs.index(channel)
            return self._task.ai_channels[ch_idx].ai_raw_samp_size

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
            return self._task.ai_channels[ch_idx].ai_dev_scaling_coeff
        else:
            return [self._task.ai_channels[idx].ai_dev_scaling_coeff for idx in ch_idx]

    def _hardware_run(self):
        """
        Todo:
            - Merge the callback creation function internally in this function.
                Since they will never be used outside of here, and only once,
                it does not make much sense to keep them as functions.
        """
        self._task = nidaqmx.Task()
        for ch in self.inputs:
            self._task.ai_channels.add_ai_voltage_chan(self.name + '/ai{}'.format(int(ch)))
        for ch in self.outputs:
            self._task.ao_channels.add_ao_voltage_chan(self.name + '/ao{}'.format(int(ch)), min_val=self.output_range[0], max_val=self.output_range[1])
        if len(self.inputs) or len(self.outputs):
            self._task.timing.cfg_samp_clk_timing(int(self.fs), samps_per_chan=self.framesize,
                    sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS)

        if self.dtype.lower() == 'unscaled' or self.dtype.lower() == 'int':
            wl = self.word_length()
            self.dtype = 'int{}'.format(int(wl))
        if len(self.inputs) > 0:
            self._task.register_every_n_samples_acquired_into_buffer_event(self.framesize, self._create_input_callback())
        if len(self.outputs):
                self._task.register_every_n_samples_transferred_from_buffer_event(self.framesize, self._create_output_callback())

        self._task.start()
        self._hardware_stop_event.wait()
        self._task.stop()
        self._task.wait_until_done(timeout=10)
        self._task.close()
        del self._task
        nidaqmx.system.Device(self.name).reset_device()

    def _create_input_callback(self):
        if self.dtype == 'int16':
            reader = nidaqmx.stream_readers.AnalogUnscaledReader(self._task.in_stream)
            databuffer = np.empty((len(self.inputs), self.framesize), dtype='int16')
            read_function = reader.read_int16
        elif self.dtype == 'int32':
            reader = nidaqmx.stream_readers.AnalogUnscaledReader(self._task.in_stream)
            databuffer = np.empty((len(self.inputs), self.framesize), dtype='int32')
            read_function = reader.read_int32
        elif self.dtype == 'uint16':
            reader = nidaqmx.stream_readers.AnalogUnscaledReader(self._task.in_stream)
            databuffer = np.empty((len(self.inputs), self.framesize), dtype='uint16')
            read_function = reader.read_uint16
        elif self.dtype == 'uint32':
            reader = nidaqmx.stream_readers.AnalogUnscaledReader(self._task.in_stream)
            databuffer = np.empty((len(self.inputs), self.framesize), dtype='uint32')
            read_function = reader.read_uint32
        else:  # Read as scaled float64
            reader = nidaqmx.stream_readers.AnalogMultiChannelReader(self._task.in_stream)
            databuffer = np.empty((len(self.inputs), self.framesize), dtype='float64')
            read_function = reader.read_many_sample

        def input_callback(task_handle, every_n_samples_event_type,
                           number_of_samples, callback_data):
            sampsRead = read_function(databuffer, self.framesize)
            self._hardware_input_Q.put(databuffer.copy())
            return 0
        return input_callback

    def _create_output_callback(self):
        writer = nidaqmx.stream_writers.AnalogMultiChannelWriter(self._task.out_stream)
        write_funciton = writer.write_many_sample
        self._task.out_stream.regen_mode = nidaqmx.constants.RegenerationMode.DONT_ALLOW_REGENERATION  # Needed to prevent issues with buffer overwrites and reuse
        write_funciton(np.zeros((self._task.out_stream.num_chans, 2*self.framesize)))  # Pre-fill the buffer with zeros, there needs to be something in the buffer when we start
        timeout = 0.5 * self.framesize / self.fs

        def output_callback(task_handle, every_n_samples_event_type,
                            number_of_samples, callback_data):
            try:
                data = self._hardware_output_Q.get(timeout=timeout)
            except queue.Empty:
                data = np.zeros((self._task.out_stream.num_chans, number_of_samples))
            sampsWritten = write_funciton(data)
            return 0
        return output_callback


class FeedbackDevice(core.Device):
    def _hardware_run(self):
        while not self._hardware_stop_event.is_set():
            try:
                self._hardware_input_Q.put(self._hardware_output_Q.get(timeout=self._hardware_timeout))
            except queue.Empty:
                pass
