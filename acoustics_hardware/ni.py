import queue
import threading
import numpy as np
import logging
from . import core

import nidaqmx
import nidaqmx.stream_readers

logger = logging.getLogger(__name__)


def getDevices(name=None):
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
    def __init__(self, name='', fs=None, framesize=10000, dtype='float64'):
        core.Device.__init__(self)
        self.name = getDevices(name)
        if fs is None:
            self.fs = nidaqmx.system.Device(self.name).ai_max_single_chan_rate
        else:
            self.fs = fs
        self.framesize = framesize  # TODO: Any automitic way to make sure that this will work? The buffer needs to be an even divisor of the device buffer size
        self.dtype = dtype
        self.inputs = []

    @property
    def max_inputchannels(self):
        return len(nidaqmx.system.Device(self.name).ai_physical_chans)

    @property
    def max_outputchannels(self):
        return len(nidaqmx.system.Device(self.name).ao_physical_chans)

    @property
    def input_range(self):
        '''
        This is only an approximate value, do NOT use for calibrating unscaled readings
        '''
        return nidaqmx.system.Device(self.name).ai_voltage_rngs

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

    def add_input(self, idx):
        # TODO: Maybe some kind af assertion that the idx is OK?
        if idx in self.inputs:
            # TODO: Raise a warning?
            return
        self.inputs.append(idx)

    def _hardware_setup(self):
        self._task = nidaqmx.Task()
        for ch in self.inputs:
            self._task.ai_channels.add_ai_voltage_chan(self.name + '/ai{}'.format(ch))
        if len(self.inputs) > 0:
            self._task.timing.cfg_samp_clk_timing(int(self.fs), samps_per_chan=self.framesize,
                    sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS)

        # TODO: Setter instead?
        if self.dtype.lower() == 'unscaled' or self.dtype.lower() == 'int':
            wl = self.word_length()
            self.dtype = 'int{}'.format(int(wl))
        if len(self.inputs) > 0:
            self._task.register_every_n_samples_acquired_into_buffer_event(self.framesize, self._create_input_callback())

    def _hardware_reset(self):
        del self._task
        nidaqmx.system.Device(self.name).reset_device()

    def _hardware_run(self):
        self._task.start()
        self._hardware_stop_event.wait()
        # TODO: How reliable is this? There have been some errors while stopping from here, but nothing that broke
        # Is it better to stop the device from within the callback?
        # It would be more expensive to check the stop event inside the callback for each frame
        self._task.stop()
        self._task.wait_until_done(timeout=10)
        self._task.close()

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


class NIDevice_old (threading.Thread):
    def __init__(self, device='', fs=None, blocksize=10000, dtype='float64'):
        threading.Thread.__init__(self)
        self.device = getDevices(device)
        if fs is None:
            self.fs = nidaqmx.system.System.local().devices[self.device].ai_max_single_chan_rate
        else:
            self.fs = fs
        self.blocksize = blocksize  # TODO: Any automitic way to make sure that this will work? The buffer needs to be an even divisor of the device buffer size
        self.dtype = dtype
        self.task = nidaqmx.Task()
        self.Q = queue.Queue()
        self.inputs = []

        # Binds the task stop function to the device stop command.
        # This works but there seems to be no possibility of waiting.
        # If the thread waits for the task using task.wait_until_done the callback will not execute while waiting, and the stop command does nothing
        # If the thread does not wait for the task, the thread finishes. The stop command still works and the task runs, but NIDevice.join will do nothing.
        # self.stop = self.task.stop
        # Binds an event set to the device stop command
        # Using this we can instead use event.wait and just stop the task afterwards.
        self._stop_event = threading.Event()
        self.stop = self._stop_event.set

        # TODO: Enable ni output devices
        # TODO: Enable ni simulaneous input/output devices
        # TODO: Enable syncronized ni devices

    @property
    def max_inputchannels(self):
        return len(nidaqmx.system.System.local().devices[self.device].ai_physical_chans)

    @property
    def max_outputchannels(self):
        return len(nidaqmx.system.System.local().devices[self.device].ao_physical_chans)

    def input_range(self, channel=None):
        '''
        This is only an approximate value, do NOT use for calibrating unscaled readings
        `channel` can be either None, which returns the overall device input range, of the
        index of a physical channel
        '''
        if channel is None:
            return nidaqmx.system.System.local().devices[self.device].ai_voltage_rngs
        ch_idx = self.inputs.index(channel)
        low = self.task.ai_channels[ch_idx].ai_rng_low
        high = self.task.ai_channels[ch_idx].ai_rng_high
        return [low, high]

    def bit_depth(self, channel=None):
        if channel is None:
            # TODO: What would be the expected behavior if the channels have different depths??
            ch_idx = 0
        else:
            ch_idx = self.inputs.index(channel)
        return self.task.ai_channels[ch_idx].ai_resolution

    def word_length(self, channel=None):
        if channel is None:
            return max([ch.ai_raw_samp_size for ch in self.task.ai_channels])
        else:
            ch_idx = self.inputs.index(channel)
            return self.task.ai_channels[ch_idx].ai_raw_samp_size

    def scaling_coeffs(self, channels=None):
        '''
        Returns the polynomial coefficients required to calculate
        input voltage from unscaled integers
        '''
        if channels is None:
            channels = self.inputs
        try:
            ch_idx = [self.inputs.index(ch) for ch in channels]
        except TypeError:
            ch_idx = self.inputs.index(channels)
            return self.task.ai_channels[ch_idx].ai_dev_scaling_coeff
        else:
            return [self.task.ai_channels[idx].ai_dev_scaling_coeff for idx in ch_idx]
            # ret_list = []
            # for ch in channels:
            #     ch_idx = self.inputs.index(ch)
            #     ret_list.append(self.task.ai_channels[ch_idx].ai_dev_scaling_coeffs)
            # return ret_list

    def add_input(self, idx, **kwargs):
        # TODO: Maybe some kind af assertion that the idx is OK?
        if idx in self.inputs:
            # TODO: Raise a warning?
            return

        self.task.ai_channels.add_ai_voltage_chan(self.device + '/ai{}'.format(idx), **kwargs)
        self.inputs.append(idx)
        # We need to configure sampling behavior as soon as possible.
        # If the sampling behavior is not configured, we will get errors when attepting to
        # access properties from the task
        self.task.timing.cfg_samp_clk_timing(int(self.fs), samps_per_chan=self.blocksize,
                sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS)

    def run(self):
        logger.info('Device started')
        self.task.timing.cfg_samp_clk_timing(int(self.fs), samps_per_chan=self.blocksize,
                sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS)
        # reader = AnalogMultiChannelReader(self.task.in_stream)
        self.blocks_read = 0  # Is not strictly neccessary but could be useful at some point, e.g. debugging
        if self.dtype.lower() == 'unscaled' or self.dtype.lower() == 'int':
            wl = self.word_length()
            self.dtype = 'int{}'.format(int(wl))
        if self.dtype == 'int16':
            reader = nidaqmx.stream_readers.AnalogUnscaledReader(self.task.in_stream)
            databuffer = np.empty((len(self.inputs), self.blocksize), dtype='int16')

            def input_callback(task_handle, every_n_samples_event_type,
                               number_of_samples, callback_data):
                sampsRead = reader.read_int16(databuffer, self.blocksize)
                self.Q.put(databuffer.copy())
                self.blocks_read += 1
                return 0
        elif self.dtype == 'int32':
            reader = nidaqmx.stream_readers.AnalogUnscaledReader(self.task.in_stream)
            databuffer = np.empty((len(self.inputs), self.blocksize), dtype='int32')

            def input_callback(task_handle, every_n_samples_event_type,
                               number_of_samples, callback_data):
                sampsRead = reader.read_int32(databuffer, self.blocksize)
                self.Q.put(databuffer.copy())
                self.blocks_read += 1
                return 0
        elif self.dtype == 'uint16':
            reader = nidaqmx.stream_readers.AnalogUnscaledReader(self.task.in_stream)
            databuffer = np.empty((len(self.inputs), self.blocksize), dtype='uint16')

            def input_callback(task_handle, every_n_samples_event_type,
                               number_of_samples, callback_data):
                sampsRead = reader.read_uint16(databuffer, self.blocksize)
                self.Q.put(databuffer.copy())
                self.blocks_read += 1
                return 0
        elif self.dtype == 'uint32':
            reader = nidaqmx.stream_readers.AnalogUnscaledReader(self.task.in_stream)
            databuffer = np.empty((len(self.inputs), self.blocksize), dtype='uint32')

            def input_callback(task_handle, every_n_samples_event_type,
                               number_of_samples, callback_data):
                sampsRead = reader.read_uint32(databuffer, self.blocksize)
                self.Q.put(databuffer.copy())
                self.blocks_read += 1
                return 0
        else:  # Read as scaled float64
            reader = nidaqmx.stream_readers.AnalogMultiChannelReader(self.task.in_stream)
            databuffer = np.empty((len(self.inputs), self.blocksize), dtype='float64')

            def input_callback(task_handle, every_n_samples_event_type,
                               number_of_samples, callback_data):
                sampsRead = reader.read_many_sample(databuffer, self.blocksize)
                self.Q.put(databuffer.copy())
                self.blocks_read += 1
                return 0
        # I think that we don't need to unregistrer callbacks before, since the thread can only run once.
        self.task.register_every_n_samples_acquired_into_buffer_event(self.blocksize, input_callback)
        # print('Starting task!')
        self.task.start()
        # print('Waiting...')
        self._stop_event.wait()
        logger.info('Device stopped')
        # print('Stopping!')
        self.task.stop()
        # print('Waiting for task...')
        self.task.wait_until_done(timeout=10)  # nidaqmx.constants.WAIT_INFINITELY
        self.task.close()  # Safety precaution, might not be nessassary
        # I got an error about not closing tasks once, so now we close them
        logger.info('Device returning!')


class NIdevice:
    def __init__(self, device=''):
        devs = getDevices()
        # Check if device is empty/none and choose the first module of the first connected device
        if device is None or len(device) == 0:
            device = devs[0]
        # Check if the specified device is a cDAQ chassis and change to the first module in said chassis.
        if device[:4] == 'cDAQ' and device[5:8] != 'Mod':
            # chassis = [dev for dev in devs if dev[:5] == device[:5]]
            # modules = [dev for dev in chassis if dev[5:8] == 'Mod']
            # device = modules[0]
            device = [dev for dev in devs if dev[:8] == device[:5] + 'Mod'][0]

        if device not in devs:
            raise ValueError('No such device available')  # TODO: Create a shared set of exceptions
        self.device = device
        self.task = CallbackTask(device)
        # We need to set these values for the task directly before we do it through the setter.
        # task.set_timing cannot run before channels have been added.
        self.task.fs = nidaqmx.system.System.local().devices[device].ai_max_single_chan_rate
        self.task.blocksize = 1000  # TODO: Make sure that this value is OK!

    @property
    def fs(self):
        return self.task.fs

    @fs.setter
    def fs(self, fs):
        self.task.set_timing(fs, self.blocksize)

    @property
    def blocksize(self):
        return self.task.blocksize

    @blocksize.setter
    def blocksize(self, blocksize):
        self.task.set_timing(self.fs, blocksize)

    @property
    def Q(self):
        return self.task.Q

    def add_input(self, idx):
        # TODO: Change to an exception
        self.task.add_input(idx)

    def start(self):
        self.task.start_acq()

    def stop(self):
        self.task.stop_acq()

    def clear_Q(self):
        # This is legacy code, should not be used in the future.
        self.input_data = np.empty((len(self.task.inputs), self.task.blocks * self.blocksize))
        block = 0
        while self.Q.unfinished_tasks > 0:
            # print(self.task.inQ.unfinished_tasks)
            data = self.Q.get()
            # data = np.zeros((len(self.task.inputs),self.blocksize))
            self.input_data[:, block * self.blocksize:(block + 1) * self.blocksize] = data
            block += 1
            self.Q.task_done()

    def __del__(self):
        self.task.close()


class CallbackTask(nidaqmx.Task):
    '''
    Extends the nidaqmx.Task class.
    '''
    def __init__(self, device):
        '''
        This class should not be created directly. Use the NIdevice class.
        '''
        # The addition of the 'device' argument sets the name of the task to the device name.
        # This could simplify debugging since the task is named, but if a new task with the name name is created before the old task is deleted, it will not work.
        # If a workflow requires recreation of the NIdevice instance, the __init__ functions will be called before the __del__ functions, and thus will throw an error.
        # nidaqmx.Task.__init__(self, device)
        nidaqmx.Task.__init__(self)
        self.device = device
        self.inputs = []
        self.outputs = []
        self.running = threading.Event()
        self.Q = queue.Queue()

    @property
    def max_inputchannels(self):
        return len(nidaqmx.system.System.local().devices[self.device].ai_physical_chans)

    @property
    def max_outputchannels(self):
        return len(nidaqmx.system.System.local().devices[self.device].ao_physical_chans)

    def set_timing(self, fs, blocksize):
        self.timing.cfg_samp_clk_timing(int(fs), samps_per_chan=blocksize,
                sample_mode=nidaqmx.constants.AcquisitionType.CONTINUOUS)
        self.fs = int(fs)
        self.blocksize = blocksize

    def add_input(self, idx):
        # TODO: Maybe some kind af assertion that the idx is OK?
        if idx in self.inputs:
            # TODO: Raise a warning?
            return

        # TODO: Make sure that the min/max values are ok for this device
        # TODO: Make the min/max values configureable at a per channel basis
        min_val = -10
        max_val = 10
        self.ai_channels.add_ai_voltage_chan(self.device + '/ai{}'.format(idx), min_val=min_val, max_val=max_val)
        self.inputs.append(idx)

    # TODO: 'add_output' and modify start method to incude outputs
    def start_acq(self):
        # TODO: Check input and output channels and modify accordningly
        # if len(self.inputs) > 0:
        self.reader = nidaqmx.stream_readers.AnalogMultiChannelReader(self.in_stream)
        self.databuffer = np.empty((len(self.inputs), self.blocksize))
        self.blocks = 0

        def input_callback(task_handle, every_n_samples_event_type,
                           number_of_samples, callback_data):
            # try:
            sampsRead = self.reader.read_many_sample(self.databuffer, self.blocksize)
            # except nidaqmx.DaqError as e:
            #   print(e)
            #   self.ex = e

            # print(sampsRead)
            self.Q.put(self.databuffer.copy())
            self.blocks += 1
            if not self.running.wait(0.1):
                # print('Timeout in callback!')
                self.stop()
                self.running.set()
            # else:
                # print('No timeout in callback...')
            return 0
        self.register_every_n_samples_acquired_into_buffer_event(self.blocksize, None)
        self.register_every_n_samples_acquired_into_buffer_event(self.blocksize, input_callback)
        # print('Setting in start_acq')
        self.running.set()
        self.start()

    def stop_acq(self):

        # print('Clearing in stop_acq!')
        self.running.clear()
        # print('Waiting in stop_acq...')
        if not self.running.wait(0.5):
            # This will execute if the acquisition thread have not stopped the task (set the event)
            # This should (hopefully) not happen and might cause exceptions if the callback runs additional times
            self.stop()
            print('Timeout in stop_acq!')
        # else:
            # print('Done waiting in stop_acq, no timeout.')
        self.running.clear()
