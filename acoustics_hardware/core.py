import queue
import threading
# import multiprocessing
import collections
import numpy as np
from . import utils
import json
from .generators import GeneratorStop


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
    """
    _generator_timeout = 1
    _trigger_timeout = 1
    _q_timeout = 1
    _hardware_timeout = 1

    def __init__(self):
        # self.input_active = multiprocessing.Event()
        self.input_active = threading.Event()
        # self.output_active = multiprocessing.Event()
        self.output_active = threading.Event()

        self.inputs = []
        self.outputs = []

        self.__generators = []
        self.__triggers = []
        self.__Qs = []
        self.__distributors = []

        # self.__main_stop_event = multiprocessing.Event()
        self.__main_stop_event = threading.Event()
        # self.__main_thread = multiprocessing.Process()
        self.__main_thread = threading.Thread()

    def start(self):
        """Starts the device.

        This creates the connections between the hardware and the software,
        configures the hardware, and initializes triggers and generators.
        Triggers are activated unless manually deactivated beforehand.
        Generators will not start generating data until the output is activated.

        Note:
            This does NOT activate inputs or outputs!
        """
        # self.__main_thread = multiprocessing.Process(target=self._Device__main_target)
        self.__main_thread = threading.Thread(target=self._Device__main_target)

        self.__main_thread.start()

    def stop(self):
        """Stops the device.

        Use this to turn off or disconnect a device safely after a measurement.
        It is not recommended to use this as deactivation control, i.e. you should
        normally not have to make multiple calls to this function.

        """
        self.__main_stop_event.set()
        # self.__process.join(timeout=10)

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
        if index not in self.outpts and index < self.max_outputs:
            self.outputs.append(Channel(index, 'output', **kwargs))

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
        """Used to flush all Qs.

        This can be useful if a measurement needs to be discarded.
        Data that have been removed from the queues, e.g. automatic file writers,
        will not be interfered with.

        Note:
            This will delete data which is still in the queues!
        """
        for q in self.__Qs:
            utils.flush_Q(q)

    def input_data(self):
        """Collects the acquired input data.

        Data in stored internally in the `Device` object while input is active.
        This method is a convenient way to access the data from a measurement
        when more elaborate and automated setups are not required.

        Returns:
            `numpy.ndarray`: Array with the input data.
            Has the shape (n_inputs, n_samples), and the input channels are
            ordered in the same order as they were added.
        """
        if self.input_active.is_set():
            print('It is not safe to get all data while input is active!')
        else:
            return utils.concatenate_Q(self.__internal_input_Q)

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
        detector = utils.LevelDetector(channel=channel, fs=self.fs, time_constant=12/frequency)
        timer = threading.Timer(interval=3, function=lambda x: self.__triggers.remove(x), args=(detector,))
        self.__triggers.append(detector)
        timer.start()
        timer.join()
        channel = self.inputs[self.inputs.index(channel)]
        channel.calibration = detector.current_level / value
        channel.unit = unit

    def _register_input_Q(self, Q=None):
        """Registers new input Q.

        This should be used to register a queue used by a Distributor. The
        queue will receive frames read while the input is active.
        For memory efficiency the input frames are not copied to individual
        queues, so in-place operations are not safe. If a Distributor needs
        to manipulate the data a copy should be made before manipulation.

        Arguments:
            Q (`~queue.Queue`, optional): The Q to register. Will be created if equal to `None`
        Returns:
            `~queue.Queue`: The registered Q.
        Note:
            The frames are NOT copied to multiple queues!
        Todo:
            Give a warning instead of an error while running.

        """
        if self.__main_thread.is_alive():
            raise UserWarning('It is not possible to register new Qs while the device is running. Stop the device and perform all setup before starting.')
        else:
            # Q = multiprocessing.Queue()
            if Q is None:
                Q = queue.Queue()
            self.__Qs.append(Q)
            return Q

    def _unregister_input_Q(self, Q):
        """Unregisters input Q.

        Removes a queue from the list of queues that receive input data.
        This method should be used by a Distributor if it is removed from
        the Device.

        Arguments:
            Q (`~queue.Queue`): The Q to remove.
        Todo:
            Give a warning instead of an error while running.
        """
        if self.__main_thread.is_alive():
            raise UserWarning('It is not possible to remove Qs while the device is running. Stop the device and perform all setup before starting.')
        else:
            self.__Qs.remove(Q)

    def add_distributor(self, distributor):
        """Adds a Distributor to the Device.

        Arguments:
            distributor: The distributor to add.
        Todo:
            Give a warning instead of an error while running.
        """
        if self.__main_thread.is_alive():
            raise UserWarning('It is not possible to add distributors while the device is running. Stop the device and perform all setup before starting.')
        else:
            self.__distributors.append(distributor)
            distributor.device = self

    def remove_distributor(self, distributor):
        """Removes a Distributor from the Device.

        Arguments:
            distributor: The distributor to remove.
        Todo:
            Give a warning instead of an error while running.
        """
        if self.__main_thread.is_alive():
            raise UserWarning('It is not possible to remove distributors while the device is running. Stop the device and perform all setup before starting.')
        else:
            self.__distributors.remove(distributor)
            try:
                distributor.remove(self)
            except AttributeError:
                distributor.device = None

    def add_trigger(self, trigger):
        """Adds a Trigger to the Device.

        Arguments:
            trigger: The trigger to add.
        Todo:
            Give a warning instead of an error while running.
        """
        if self.__main_thread.is_alive():
            raise UserWarning('It is not possible to add new triggers while the device is running. Stop the device and perform all setup before starting.')
        else:
            self.__triggers.append(trigger)
            trigger.device = self

    def remove_trigger(self, trigger):
        """Removes a Trigger from the Device.

        Arguments:
            trigger: The trigger to remove.
        Todo:
            Give a warning instead of an error while running.
        """
        if self.__main_thread.is_alive():
            raise UserWarning('It is not possible to remove triggers while the device is running. Stop the device and perform all setup before starting.')
        else:
            self.__triggers.remove(trigger)
            trigger.device = None

    def add_generator(self, generator):
        """Adds a Generator to the Device.

        Arguments:
            generator: The generator to add.
        Note:
            The order that multiple generators are added to a device
            dictates which output channel receives data from which generator.
            The total number of generated channels must match the number of
            output channels.
        Todo:
            Give a warning instead of an error while running.
        """
        if self.__main_thread.is_alive():
            raise UserWarning('It is not possible to add new generators while the device is running. Stop the device and perform all setup before starting.')
        else:
            self.__generators.append(generator)
            generator.device = self

    def remove_generator(self, generator):
        """Removes a Generator from the Device.

        Arguments:
            generator: The generator to remove.
        Todo:
            Give a warning instead of an error while running.
        """
        if self.__main_thread.is_alive():
            raise UserWarning('It is not possible to add new generators while the device is running. Stop the device and perform all setup before starting.')
        else:
            self.__generators.remove(generator)
            generator.device = None

    def __reset(self):
        """Resets the `Device`.

        Performs a number of tasks required to restart a device after it has
        been stopped:

        - Clears all stop events
        - Resets all triggers
        - Resets all generators
        - Clears input and output activation

        """
        self.__main_stop_event.clear()
        self.__trigger_stop_event.clear()
        self.__q_stop_event.clear()
        self._hardware_stop_event.clear()
        for trigger in self.__triggers:
            trigger.reset()
        for generator in self.__generators:
            generator.reset()
        self.input_active.clear()
        self.output_active.clear()

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
        self._hardware_input_Q = queue.Queue()
        self._hardware_output_Q = queue.Queue(maxsize=25)
        self._hardware_stop_event = threading.Event()
        self.__triggered_q = queue.Queue()
        self.__internal_input_Q = queue.Queue()
        self.__Qs.append(self.__internal_input_Q)
        self.__generator_stop_event = threading.Event()
        self.__trigger_stop_event = threading.Event()
        self.__q_stop_event = threading.Event()
        # Start hardware in separate thread
        # Manage triggers in separate thread
        # Manage Qs in separate thread
        generator_thread = threading.Thread(target=self.__generator_target)
        hardware_thread = threading.Thread(target=self._hardware_run)
        trigger_thread = threading.Thread(target=self.__trigger_target)
        q_thread = threading.Thread(target=self.__q_target)

        generator_thread.start()
        hardware_thread.start()
        trigger_thread.start()
        q_thread.start()

        self.__main_stop_event.wait()

        self.__generator_stop_event.set()
        generator_thread.join()
        self._hardware_stop_event.set()
        hardware_thread.join()
        self.__trigger_stop_event.set()
        trigger_thread.join()
        self.__q_stop_event.set()
        q_thread.join()
        self.__reset()

    def __trigger_target(self):
        """Trigger handling method.

        This method will execute as a subthread in the device, responsible
        for managing the attached triggers, and handling input data.

        Todo:
            - Pre-triggering using appropriate values
            - Post-triggering
            - Aligning triggers?
        """
        data_buffer = collections.deque(maxlen=10)
        for trigger in self.__triggers:
            trigger.setup()

        while not self.__trigger_stop_event.is_set():
            # Wait for a frame, if none has arrived within the set timeout, go back and check stop condition
            try:
                this_frame = self._hardware_input_Q.get(timeout=self._trigger_timeout)
            except queue.Empty:
                continue
            # Execute all triggering conditions
            scaled_frame = self._input_scaling(this_frame)
            for trig in self.__triggers:
                trig(scaled_frame)
            # Move the frame to the buffer
            data_buffer.append(this_frame)
            # If the trigger is active, move everything from the data buffer to the triggered Q
            if self.input_active.is_set():
                while len(data_buffer) > 0:
                    self.__triggered_q.put(data_buffer.popleft())

        # The hardware should have stopped by now, analyze all remaining frames.
        while True:
            try:
                this_frame = self._hardware_input_Q.get(timeout=self._trigger_timeout)
            except queue.Empty:
                break
            scaled_frame = self._input_scaling(this_frame)
            for trig in self.__triggers:
                trig(scaled_frame)
            data_buffer.append(this_frame)
            if self.input_active.is_set():
                while len(data_buffer) > 0:
                    self.__triggered_q.put(data_buffer.popleft())

    def __q_target(self):
        """Queue handling method.

        This method will execute as a subthread in the device, responsible
        for moving data from the input (while active) to the queues used by
        Distributors.
        """
        for distributor in self.__distributors:
            distributor.setup()

        while not self.__q_stop_event.is_set():
            # Wait for a frame, if none has arrived within the set timeout, go back and check stop condition
            try:
                this_frame = self.__triggered_q.get(timeout=self._q_timeout)
            except queue.Empty:
                continue
            for Q in self.__Qs:
                # Copy the frame to all output Qs
                Q.put(this_frame)

        # The triggering should have stopped by now, move the remaining frames.
        while True:
            try:
                this_frame = self.__triggered_q.get(timeout=self._q_timeout)
            except queue.Empty:
                break
            for Q in self.__Qs:
                Q.put(this_frame)

    def __generator_target(self):
        """Generator handling method.

        This method will execute as a subthread in the device, responsible
        for managing attached generators, and generating output frames from
        the generators.
        """
        for generator in self.__generators:
            generator.setup()
        use_prev_frame = False
        while not self.__generator_stop_event.is_set():
            if self.output_active.wait(timeout=self._generator_timeout):
                try:
                    if not use_prev_frame:
                        frame = np.concatenate([generator() for generator in self.__generators])
                except GeneratorStop:
                    self.output_active.clear()
                    continue
                try:
                    self._hardware_output_Q.put(frame, timeout=self._generator_timeout)
                except queue.Full:
                    use_prev_frame = True


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
