from queue import Queue
from threading import Event, current_thread
import numpy as np

import nidaqmx
from nidaqmx.stream_readers import AnalogMultiChannelReader


def getAvailableDevices():
    system = nidaqmx.system.System.local()
    return [dev.name for dev in system.devices]


class NIdevice:
    def __init__(self, device=''):
        devs = getAvailableDevices()
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
        self.task.fs = nidaqmx.system.System.local().devices[device].ai_max_single_chan_rate
        self.blocksize = 1000  # TODO: Make sure that this value is OK!

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
        self.running = Event()
        self.Q = Queue()

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
        self.reader = AnalogMultiChannelReader(self.in_stream)
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
