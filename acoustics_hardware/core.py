import warnings
import queue
import threading
# import multiprocessing
import collections
import numpy as np
import scipy.signal
from . import utils
import json
import logging
from .generators import GeneratorStop
from .distributors import QDistributor


logger = logging.getLogger(__name__)

class Device:
    """Abstract class that provides a consistent framework for different hardware.

    An instance of a specific implementation of of `Device` is typically linked
    to a single physical input/output device. The instance manages connections to
    the device, any attached triggers, enabling/disabling input/output, and
    attached generators.

    Attributes:
        input_active (`~threading.Event`): Controls input state.
            Use `~threading.Event.set` to activate input and `~threading.Event.clear` to deactivate.
        output_active (`~threading.Event`): Controls output state.
            Use `~threading.Event.set` to activate output and `~threading.Event.clear` to deactivate.
        inputs (list[`Channel`]): List of assigned inputs, see `add_input`.
        outputs (list[`Channel`]): List of assigned outputs, see `add_output`.
        max_inputs (`int`): The maximum number of inputs available.
        max_outputs (`int`): The maximum number of outputs available.
        calibrations (`numpy.ndarray`): Calibrations of input channels, defaults to 1 for missing calibrations.

    Todo:
        Remove the `add_trigger`, `remove_trigger`, `add_distributor`, `remove_distributor` and possibly
        `add_generator` and `remove_generator`. Since these other objets will (almost) always have a single device
        which devines the input data for the object it would be reasonable to have said device as a property of that
        object. The code to manage adding/removing from the device will then be implenented in those objects. This
        wil in the long run reduce the number of calls, since the device can be given as an input argument when creating
        the objects. It also allows subclasses to customize how the objects are added to the device.

    """
    __device_count = 0
    _generator_timeout = 1e-3
    _trigger_timeout = 1e-3
    _q_timeout = 1e-3
    _hardware_timeout = 1e-3

    def __init__(self, **kwargs):
        # self.input_active = multiprocessing.Event()
        self.input_active = threading.Event()
        # self.output_active = multiprocessing.Event()
        self.output_active = threading.Event()

        self.inputs = []
        self.outputs = []

        self.__generators = []
        self.__triggers = []
        self.__output_triggers = []
        self.__distributors = []

        # self.__main_stop_event = multiprocessing.Event()
        self.__main_stop_event = threading.Event()
        # self.__main_thread = multiprocessing.Process()
        self.__main_thread = threading.Thread()

        self.__name = 'Device_{}'.format(Device.__device_count)
        Device.__device_count += 1

        kwargs.setdefault('fs', 1)  # This is required for all devices
        kwargs.setdefault('framesize', 1)
        for key, value in kwargs.items():
            setattr(self, key, value)

    def initialize(self):
        """Initializes the device.

        This creates the connections between the hardware and the software,
        configures the hardware, and initializes triggers and generators.
        Triggers are activated unless manually deactivated beforehand.
        Generators will not start generating data until the output is activated.

        Note:
            This does NOT activate inputs or outputs!
        """
        # self.__main_thread = multiprocessing.Process(target=self._Device__main_target)
        try:
            name = self.name + ' (' + self.__name + ')'
        except AttributeError:
            name = self.__name
        self.__main_thread = threading.Thread(target=self._Device__main_target, name=name)

        self.__main_thread.start()

    def terminate(self):
        """Terminates the device.

        Use this to turn off or disconnect a device safely after a measurement.
        It is not recommended to use this as deactivation control, i.e. you should
        normally not have to make multiple calls to this function.

        """
        self.__main_stop_event.set()
        # self.__process.join(timeout=10)

    def start(self, timed=False, input=True, output=True, blocking=False):
        if not self.initialized:
            self.initialize()
        if timed:
            timer = threading.Timer(interval=timed, function=self.stop, kwargs={"input":input, "output":output})
            timer.start()

        if input:
            self.input_active.set()
        if output:
            self.output_active.set()
        if blocking:
            timer.join()

    def stop(self, input=True, output=True):
        if input:
            self.input_active.clear()
        if output:
            self.output_active.clear()

    def add_input(self, index, **kwargs):
        """Adds a new input `Channel`.

        Arguments:
            index (`int`): Zero-based index of the channel.
            **kwargs: All arguments of `Channel` except ``chtype`` and ``index``.

        """
        if index not in self.inputs and index < self.max_inputs:
            self.inputs.append(Channel(index, 'input', **kwargs))

    def add_output(self, index, **kwargs):
        """Adds a new output `Channel`.

        Arguments:
            index (`int`): Zero-based index of the channel.
            **kwargs: All arguments of `Channel` except ``chtype`` and ``index``.
        """
        if index not in self.outputs and index < self.max_outputs:
            self.outputs.append(Channel(index, 'output', **kwargs))

    @property
    def pre_triggering(self):
        try:
            return self._pre_triggering
        except AttributeError:
            self._pre_triggering = 0
            return self.pre_triggering

    @pre_triggering.setter
    def pre_triggering(self, val):
        if self.__main_thread.is_alive():
            raise UserWarning('It is not possible to change the pre-triggering time while the device is running. Stop the device and perform all setup before starting.')
        else:
            self._pre_triggering = val

    @property
    def post_triggering(self):
        try:
            return self._post_triggering
        except AttributeError:
            self._post_triggering = 0
            return self.post_triggering

    @post_triggering.setter
    def post_triggering(self, val):
        if self.__main_thread.is_alive():
            raise UserWarning('It is not possible to change the post-triggering time while the device is running. Stop the device and perform all setup before starting.')
        else:
            self._post_triggering = val

    @property
    def calibrations(self):
        return np.array([c.calibration if c.calibration is not None else 1 for c in self.inputs])

    def _input_scaling(self, frame):
        """Scales the input for triggering

        This is separate from applying calibration values, which is controlled
        by each trigger. The intention here is to account for data formats,
        e.g. reading data as int32 instead of floats or removing DC offsets.

        Arguments:
            frame (`numpy.ndarray`): Unscaled input frame.
        Returns:
            `numpy.ndarray`: Scaled input frame
        """
        return frame

    @property
    def max_inputs(self):
        return np.inf

    @property
    def max_outputs(self):
        return np.inf

    def _hardware_run(self):
        """This is the primary method in which hardware interfacing should be implemented.

        This method will run in a separate thread. It is responsible for creating the
        connections to the hardware according to the configurations.
        If the device has registered inputs, they should always be read and frames
        should be put in the input Q (in the order registered).
        If the device has registered outputs, frames should be taken from the output Q
        and and output through the physical channels (in the order registered).
        If a specific hardware requires constant data streams, fill the stream
        with zeros if no output data is available. The timings for this must be
        implemented by this method.

        This method is responsible for checking if the stop event is set with the interval
        specified by the hardware timeout attribute. When this thread receives the signal
        to stop, it must close all connections to the device and reset it to a state from
        where it can be started again (possibly from another instance).

        Attributes:
            _hardware_input_Q (`~queue.Queue`): The Q where read input data should go.
            _hardware_output_Q (`~queue.Queue`): The Q where the output data to write is stored.
            _hardware_stop_event (`~threading.Event`): Tells the hardware thread to stop.
            _hardware_timeout (`float`): How often the stop event should be checked.
        """
        raise NotImplementedError('Required method `_hardware_run` not implemented in {}'.format(self.__class__.__name__))

    def flush(self):
        """Used to flush the internal data.

        This can be useful if a measurement needs to be discarded.
        Data that have been removed from the queues, e.g. automatic file writers,
        will not be interfered with.

        Note:
            This will delete data which is still in the stored!
        """
        self.__internal_distributor.flush()

    def get_input_data(self, blocking=True, timeout=-1):
        """Collects the acquired input data.

        Data in stored internally in the `Device` object while input is active.
        This method is a convenient way to access the data from a measurement
        when more elaborate and automated setups are not required.

        Returns:
            `numpy.ndarray`: Array with the input data.
            Has the shape (n_inputs, n_samples), and the input channels are
            ordered in the same order as they were added.
        """
        do_relese = self.__input_data_lock.acquire(blocking=blocking, timeout=timeout)
        try:
            data = self.__internal_distributor.data
        except AttributeError:
            raise ValueError('Cannot get input data from uninitialized device!')
        except ValueError:
            raise ValueError('No input data to get!')
        else:
            return data
        finally:
            if do_relese:
                self.__input_data_lock.release()

    def calibrate(self, channel, frequency=1e3, value=1, ctype='rms', unit='V'):
        """Calibrates a channel using a reference signal.

        The resulting calibration value and unit is stored as attributes of the
        corresponding `Channel`. Different calibration types should be used for different
        instruments.
        Currently only unfiltered RMS calibrations are implemented. This detects the level
        in the signal for 3 seconds, and uses the final level as the calibration value.

        Arguments:
            channel (`int`): Index of the channel, in the order that they were added to the device.
            frequency (`float`): The frequency of the applied reference signal, defaults to 1 kHz.
            value (`float`): The value of the reference signal, defaults to 1.
            ctype (``'rms'``): Use to switch between different calibration methods. Currently not used.
            unit (`str`): The unit of the calibrated quantity, defaults to ``'V'``.
        Todo:
            - Filtering the input before detecting the level
            - Average over multiple parts
            - Determine a reasonable value of the averaging coefficient
        """
        q = QDistributor()
        timer = threading.Timer(interval=3, function=lambda x: self.__triggers.remove(x), args=(q,))
        self.__triggers.append(q)
        timer.start()
        timer.join()

        wn = frequency * np.array([0.9, 1.1]) / self.fs * 2
        sos = scipy.signal.iirfilter(8, wn, output='sos')
        data = q.data[channel]
        data = scipy.signal.sosfilt(sos, data)

        channel = self.inputs[self.inputs.index(channel)]
        channel.calibration = np.std(data) / value
        channel.unit = unit

    @property
    def initialized(self):
        return self.__main_thread.is_alive()

    def reset(self, triggers=True, generators=True, distributors=True):
        """Resets the `Device`.

        Resets the attached objets of the device. Note that triggers will reset
        the triggers which are attached to this device, which may or may not be the
        same triggers used to trigger this device.

        Arguments:
            triggers (`bool`): Whether to reset the triggers, default True.
            generators (`bool`): Whether to reset the generators, default True.
            distributors (`bool`): Whether to reset the distributors, default True.

        """
        self.__main_stop_event.clear()
        self._hardware_stop_event.clear()
        if triggers:
            for trigger in self.__triggers:
                trigger.reset()
        if generators:
            for generator in self.__generators:
                generator.reset()
        if distributors:
            for distributor in self.__distributors:
                distributor.reset()

    def _Device__main_target(self):
        """Main method for a Device.

        This is the method that is executed when the device is started.
        Four other threads will be started in this method, one for generators,
        one for triggers, one for queue handling, and one for the hardware.
        This method is also responsible for creating most of the queues and
        events that connect the other threads, as well as initializing the
        distributors.
        """
        # The explicit naming of this method is needed on windows for some stange reason.
        # If we rely on the automatic name wrangling for the process target, it will not be found in device subclasses.
        logger.info('Device initializing')
        self._hardware_input_Q = queue.Queue()
        self._hardware_output_Q = MasterSlaveQueue(maxsize=5)
        self.__input_data_lock = threading.Lock()
        self._hardware_stop_event = threading.Event()
        self.__triggered_q = queue.Queue()
        try:
            if self.__internal_distributor is None:
                raise AttributeError
        except AttributeError:
            with warnings.catch_warnings() as w:
                warnings.filterwarnings('ignore', module='acoustics_hardware.distributors')
                self.__internal_distributor = QDistributor(device=self)

        self.__generator_stop_event = threading.Event()

        # Start hardware in separate thread
        # Manage triggers in separate thread
        # Manage Qs in separate thread
        name = self.__main_thread.name
        generator_thread = threading.Thread(target=self.__generator_target, name=name + ' - generator')
        output_trigger_thread = threading.Thread(target=self.__output_trigger_target, name=name + ' - outputtrigger')
        hardware_thread = threading.Thread(target=self._hardware_run, name=name + ' - hardware')
        trigger_thread = threading.Thread(target=self.__trigger_target, name=name + ' - trigger')
        distributor_thread = threading.Thread(target=self.__distributor_target, name=name + ' - distributor')
        generator_thread.start()
        output_trigger_thread.start()
        hardware_thread.start()
        trigger_thread.start()
        distributor_thread.start()

        logger.verbose('Device initialized')
        self.__main_stop_event.wait()
        self._hardware_stop_event.set()
        hardware_thread.join()

        self.__generator_stop_event.set()
        generator_thread.join()
        output_trigger_thread.join()

        self._hardware_input_Q.put(False)
        trigger_thread.join()
        distributor_thread.join()
        self.stop()
        self.reset()
        logger.verbose('Device terminated')

    def __trigger_target(self):
        """Trigger handling method.

        This method will execute as a subthread in the device, responsible
        for managing the attached triggers, and handling input data.

        Todo:
            - Make sure that the sample level triggering works on real hardware
            - Process the remaining frames after the hardware thread has stopped
            - Alignment is a temporary fix, and will not work across multiple devices
        """
        pre_trigger_samples = int(np.ceil(self.pre_triggering * self.fs))
        post_trigger_samples = int(np.ceil(self.post_triggering * self.fs))
        remaining_samples = 0
        data_buffer = collections.deque(maxlen=pre_trigger_samples // self.framesize + 2)
        triggered = False
        collecting_input = False
        self._trigger_alignment = 0
        self.__hardware_input_frames = 0
        self.__triggered_frames = 0

        for trigger in self.__triggers:
            trigger.setup()

        logger.verbose('Triggers running')
        while True:
            # Wait for a frame, if none has arrived within the set timeout, go back and check stop condition
            try:
                this_frame = self._hardware_input_Q.get(timeout=self._trigger_timeout)
            except queue.Empty:
                continue
            if this_frame is False:
                # Stop signal
                self._hardware_input_Q.task_done()
                break
            # Execute all triggering conditions
            scaled_frame = self._input_scaling(this_frame)
            for trig in self.__triggers:
                trig(scaled_frame)
            self._hardware_input_Q.task_done()
            self.__hardware_input_frames += 1

            # Move the frame to the buffer
            data_buffer.append(this_frame)
            # If the trigger is active, move everything from the data buffer to the triggered Q
            if self.input_active.is_set() and not triggered:
                logger.debug('Input activated')
                collecting_input = self.__input_data_lock.acquire()
                # Triggering happened between this frame and the last, do pre-prigger aligniment
                triggered = True
                trigger_sample_index = int(self._trigger_alignment * self.fs) + (len(data_buffer) - 1) * self.framesize - pre_trigger_samples
                while trigger_sample_index > 0:
                    if trigger_sample_index >= self.framesize:
                        data_buffer.popleft()
                    else:
                        self.__triggered_q.put(data_buffer.popleft()[..., trigger_sample_index:])
                    trigger_sample_index -= self.framesize
                remaining_samples = len(data_buffer) * self.framesize
            elif self.input_active.is_set():
                # Continue moving data to triggered Q
                remaining_samples += self.framesize
            elif not self.input_active.is_set() and triggered:
                logger.debug('Input deactivated')
                # Just detriggered, set remaining samples correctly
                triggered = False
                remaining_samples = post_trigger_samples + int(self._trigger_alignment * self.fs) + 1

            while remaining_samples > 0:
                try:
                    frame = data_buffer.popleft()
                except IndexError:
                    break
                self.__triggered_q.put(frame[..., :remaining_samples])
                self.__triggered_frames += 1
                remaining_samples -= frame.shape[-1]
            else:
                if collecting_input and not triggered:
                    self.__input_data_lock.release()
                    collecting_input = False

        if self.__input_data_lock.locked():
            self.__input_data_lock.release()
        self.__triggered_q.put(False)  # Signal the q-handler thread to stop

    def __distributor_target(self):
        """Queue handling method.

        This method will execute as a subthread in the device, responsible
        for moving data from the input (while active) to the queues used by
        Distributors.

        Todo:
            Redo the framework for Qs and distributors. If Distributors are made
            callable, we would just call all distributors with the frame. If a
            distributors needs a Q and runs some expensive processing in a different
            thread, is is easy to implement a call function for taking the frame
            and putting it in a Q owned by the distributor.
        """
        for distributor in self.__distributors:
            distributor.setup()
        self.__distributed_frames = 0

        logger.verbose('Distributors running')
        while True:
            # Wait for a frame, if none has arrived within the set timeout, go back and check stop condition
            try:
                this_frame = self.__triggered_q.get(timeout=self._q_timeout)
            except queue.Empty:
                continue
            for distributor in self.__distributors:
                # Copy the frame to all output Qs
                distributor(this_frame)
            self.__triggered_q.task_done()
            self.__distributed_frames += 1
            if this_frame is False:
                # Signal to stop, we have sent it to all distributors if they need it
                break

    def __generator_target(self):
        """Generator handling method.

        This method will execute as a subthread in the device, responsible
        for managing attached generators, and generating output frames from
        the generators.
        """
        for generator in self.__generators:
            generator.setup()
        self.__generated_frames = 0
        self.__hardware_output_frames = 0
        generating = False

        use_prev_frame = False
        logger.verbose('Generators running')
        while not self.__generator_stop_event.is_set():
            if self.output_active.is_set():
                if not generating:
                    # First frame generating
                    generating = True
                    logger.debug('Output activated')
                try:
                    if not use_prev_frame:
                        frame = np.concatenate([generator() for generator in self.__generators])
                        self.__generated_frames += 1
                except (GeneratorStop, ValueError):
                    logger.debug('Generator halted output')
                    self.output_active.clear()
                    continue
            else:
                if generating:
                    # First frame not generating
                    generating = False
                    logger.debug('Output deactivated')
                frame = np.zeros((len(self.outputs), self.framesize))
            try:
                self._hardware_output_Q.put(frame, timeout=self._generator_timeout)
            except queue.Full:
                use_prev_frame = True
            else:
                use_prev_frame = False
                self.__hardware_output_frames += 1

        # Clear out the output Q to halt the output trigger thread
        while True:
            try:
                self._hardware_output_Q.get_nowait()
                self._hardware_output_Q.task_done()
                self.__hardware_output_frames += 1
            except queue.Empty:
                break
        self._hardware_output_Q.put(False)
        self._hardware_output_Q.task_done()

    def __output_trigger_target(self):
        for trig in self.__output_triggers:
            trig.setup()

        self.__output_triggered_frames = 0
        logger.verbose('Output triggers running')
        while True:
            frame = self._hardware_output_Q.get_slave()
            if frame is False:
                self._hardware_output_Q.slave_task_done()
                break
            for trig in self.__output_triggers:
                trig(frame)
            self._hardware_output_Q.slave_task_done()
            self.__output_triggered_frames += 1


class MasterSlaveQueue(queue.Queue):
    def __init__(self, *args, slaves=1, **kwargs):
        super().__init__(*args, **kwargs)

        self._slave = queue.Queue()
        self._counter = threading.Semaphore(0)

    def task_done(self):
        super().task_done()
        self._counter.release()

    def slave_task_done(self):
        self._slave.task_done()

    def get_slave(self, block=True, timeout=None):
        self._counter.acquire(blocking=block, timeout=timeout)
        return self._slave.get_nowait()

    def put(self, item, block=True, timeout=None):
        super().put(item, block=block, timeout=timeout)
        self._slave.put(item)


class Channel(int):
    """Represents a channel of a device.

    Contains information about a physical channel used.

    Arguments:
        index (`int`): Zero-based index of the channel in the device.
        chtype (``'input'`` or ``'output'``): Type of channel.
        label (`str`, optional): User label for identification of the channel.
        calibration (`float`, optional): Manual calibration value.
        unit (`str`, optional): Physical unit of the calibrated channel.
    """
    @classmethod
    def from_json(cls, json_dict):
        """Creates a channel from json representation.

        Arguments:
            json_dict (`str`): json representation of a dictionary containing
                key-value pairs for the arguments of a `Channel`.
        Returns:
            `Channel`: A channel with the given specification.
        """
        return cls(**json.loads(json_dict))

    def to_json(self):
        """Create json representation of this channel.

        Returns:
            `str`: json representation.
        """
        return json.dumps(self.__dict__)

    def __new__(cls, index, *args, **kwargs):
        return super(Channel, cls).__new__(cls, index)

    def __init__(self, index, chtype, label=None, calibration=None, unit=None):
        self.index = index
        self.chtype = chtype
        self.label = label
        self.calibration = calibration
        self.unit = unit

    def __repr__(self):
        return self.__class__.__name__ + '(' + ', '.join(['{}={}'.format(key, value) for key, value in self.__dict__.items()]) + ')'

    def __str__(self):
        label_str = '' if self.label is None else ' "{}"'.format(self.label)
        calib_str = '' if self.calibration is None else ' ({:.4g} {})'.format(self.calibration, self.unit)
        ch_str = '{chtype} channel {index}'.format(chtype=self.chtype, index=self.index).capitalize()
        return '{ch}{label}{calib}'.format(ch=ch_str, label=label_str, calib=calib_str)
