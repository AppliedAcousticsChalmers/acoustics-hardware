import queue
import numpy as np
import logging
from . import core

import nidaqmx
import nidaqmx.stream_readers
import nidaqmx.stream_writers

logger = logging.getLogger(__name__)


def get_devices(name=None):
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


class NIDevice(core.Device):
    # TODO: Implement output devices. Caveat: We would need an output device to test the implementation...
    def __init__(self, name=None, fs=None, framesize=10000, dtype='float64'):
        core.Device.__init__(self)
        if name is None:
            name = get_devices()[0]
        self.name = get_devices(name)
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
        self.framesize = framesize  # TODO: Any automitic way to make sure that this will work? The buffer needs to be an even divisor of the device buffer size
        # The device buffer size can be accessed via the task in/out stream, but these are tricky to access.
        self.dtype = dtype

    @property
    def max_inputs(self):
        return len(nidaqmx.system.Device(self.name).ai_physical_chans)

    @property
    def max_outputs(self):
        return len(nidaqmx.system.Device(self.name).ao_physical_chans)

    @property
    def input_range(self):
        '''
        This is only an approximate value, do NOT use for calibrating unscaled readings
        '''
        return nidaqmx.system.Device(self.name).ai_voltage_rngs

    @property
    def output_range(self):
        return nidaqmx.system.Device(self.name).ao_voltage_rngs

    def bit_depth(self, channel=None):
        # TODO: This can only be called from the device process, otherwise the task is not available
        if channel is None:
            # TODO: What would be the expected behavior if the channels have different depths??
            ch_idx = 0
        else:
            ch_idx = self.inputs.index(channel)
        return self._task.ai_channels[ch_idx].ai_resolution

    def word_length(self, channel=None):
        # TODO: This can only be called from the device process, otherwise the task is not available
        if channel is None:
            return max([ch.ai_raw_samp_size for ch in self._task.ai_channels])
        else:
            ch_idx = self.inputs.index(channel)
            return self._task.ai_channels[ch_idx].ai_raw_samp_size

    def scaling_coeffs(self, channels=None):
        '''
        Returns the polynomial coefficients required to calculate
        input voltage from unscaled integers
        '''
        # TODO: This can only be called from the device process, otherwise the task is not available
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
        self._task = nidaqmx.Task()
        for ch in self.inputs:
            self._task.ai_channels.add_ai_voltage_chan(self.name + '/ai{}'.format(int(ch)))
        for ch in self.outputs:
            self._task.ao_channels.add_ao_voltage_chan(self.name + '/ao{}'.format(int(ch)), min_val=self.output_range[0], max_val=self.output_range[1])
        if len(self.inputs) or len(self.outputs):
            self._task.timing.cfg_samp_clk_timing(int(self.fs), samps_per_chan=self.framesize,
                    sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS)

        # TODO: Setter instead?
        if self.dtype.lower() == 'unscaled' or self.dtype.lower() == 'int':
            wl = self.word_length()
            self.dtype = 'int{}'.format(int(wl))
        if len(self.inputs) > 0:
            self._task.register_every_n_samples_acquired_into_buffer_event(self.framesize, self._create_input_callback())
        if len(self.outputs):
                self._task.register_every_n_samples_transferred_from_buffer_event(self.framesize, self._create_output_callback())

        self._task.start()
        self._hardware_stop_event.wait()
        # TODO: How reliable is this? There have been some errors while stopping from here, but nothing that broke
        # Is it better to stop the device from within the callback?
        # It would be more expensive to check the stop event inside the callback for each frame
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
