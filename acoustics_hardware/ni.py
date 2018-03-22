import queue
import numpy as np
import logging
from . import core

try:
    import nidaqmx
    import nidaqmx.stream_readers
    import nidaqmx.stream_writers
except ModuleNotFoundError:
    pass

logger = logging.getLogger(__name__)


class NIDevice(core.Device):
    """Class for interacting with national instruments hardware.

    Implementation of the `~core.Device` framework for national instruments hardware.
    Built on top of the `nidaqmx <https://nidaqmx-python.readthedocs.io/>`_ package.

    See Also:
        `acoustics_hardware.core.Device`.
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
        system = nidaqmx.system.System.local()
        name_list = [dev.name for dev in system.devices]
        if name is None:
            return name_list
        else:
            if len(name) == 0:
                name = name_list[0]
            if name[:4] == 'cDAQ' and name[5:8] != 'Mod':
                name = [x for x in name_list if x[:8] == name[:5] + 'Mod'][0]
            return name

    def __init__(self, name=None, fs=None, framesize=10000, dtype='float64'):
        core.Device.__init__(self)
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
