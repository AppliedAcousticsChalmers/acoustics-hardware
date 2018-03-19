import queue
import threading
# import multiprocessing
import collections
import numpy as np
from . import utils
import json


class Device:
    """
    Top level abstract Device class used for inheritance to implement specific hardware.
    Required methods to implement:
    - `_hardware_run`: Main method used to run the hardware

    Optional methods to override:
    - `_hardware_stop`: Use to stop the hardware
    - `_hardware_reset`: Use to reset the hardware/python object to a startable state

    See the documentation of these for more information.

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

        # self.__main_stop_event = multiprocessing.Event()
        self.__main_stop_event = threading.Event()
        # self.__main_thread = multiprocessing.Process()
        self.__main_thread = threading.Thread()

    def start(self):
        # TODO: Documentation
        # self.__main_thread = multiprocessing.Process(target=self._Device__main_target)
        self.__main_thread = threading.Thread(target=self._Device__main_target)

        self.__main_thread.start()

    def stop(self):
        # TODO: Documentation
        self.__main_stop_event.set()
        # self.__process.join(timeout=10)
        # TODO: We will not wait for the process now, since it will not finish if there are
        # items left in the Q.

    def add_input(self, index, **kwargs):
        if index not in self.inputs and index < self.max_inputs:
            self.inputs.append(Channel(index, 'input', **kwargs))

    def add_output(self, index, **kwargs):
        if index not in self.outpts and index < self.max_outputs:
            self.outputs.append(Channel(index, 'output', **kwargs))

    @property
    def calibrations(self):
        return np.array([c.calibration if c.calibration is not None else 1 for c in self.inputs])

    def input_scaling(self, frame):
        return frame

    @property
    def max_inputs(self):
        return np.inf

    @property
    def max_outputs(self):
        return np.inf

    def _hardware_run(self):
        '''
        This is the primary method in which hardware interfacing should be implemented.
        There are two Qs, two required events, and two default controll events.
            `_hardware_input_Q`: The Q where read input data should go
            `_hardware_output_Q`: The Q where the output data to write is stored
            `input_active`: The event which toggles if data should be placed in the input Q
            `output_active`; The event whoch toggles if data should be read from the output Q
            `_hardware_stop_event`: Tells the hardware thread to stop, see `_hardware_stop` and `_hardware_reset`
        '''
        raise NotImplementedError('Required method `_hardware_run` not implemented in {}'.format(self.__class__.__name__))

    def _hardware_stop(self):
        '''
        This is the primary method used for stopping the hardware.
        It is reccomended to do this using Events inside the _hardware_run method.

        Default implementation:
            self._hardware_stop_event.set()
        '''
        self._hardware_stop_event.set()

    def _hardware_reset(self):
        '''
        This method works in combination with `_hardware_setup` to put the hardware back to a state
        where it can be started again.
        '''
        pass

    def flush(self):
        '''
        Used to flush all Qs so that processes can terminate.
        THIS WILL DELETE DATA which is still in the Qs
        '''
        for q in self.__Qs:
            utils.flush_Q(q)

    def calibrate(self, channel, frequency=1e3, value=1, type='rms', unit='V'):
        # TODO: Is this a good value for the time_constant?
        detector = utils.LevelDetector(channel=channel, fs=self.fs, time_constant=12/frequency)
        timer = threading.Timer(interval=3, function=lambda x: self.__triggers.remove(x), args=(detector,))
        self.__triggers.append(detector)
        timer.start()
        timer.join()
        channel = self.inputs[self.inputs.index(channel)]
        channel.calibration = detector.current_level / value
        channel.unit = unit

    @property
    def input_Q(self):
        if self.__main_thread.is_alive():
            # TODO: Custom warning class
            raise UserWarning('It is not possible to register new Qs while the device is running. Stop the device and perform all setup before starting.')
        else:
            # Q = multiprocessing.Queue()
            Q = queue.Queue()
            self.__Qs.append(Q)
            return Q

    def remove_Q(self, Q):
        if self.__main_thread.is_alive():
            # TODO: Custom warning class
            raise UserWarning('It is not possible to remove Qs while the device is running. Stop the device and perform all setup before starting.')
        else:
            # TODO: What should happen if the Q is not in the list?
            self.__Qs.remove(Q)

    @property
    def output_Q(self):
        # TODO: Documentation
        # This only exists to have consistent naming conventions (_hardware_input_Q, _hardware_output_Q) for subclassing
        return self._hardware_output_Q

    def add_trigger(self, trigger):
        # TODO: Documentation
        if self.__main_thread.is_alive():
            # TODO: Custom warning class
            raise UserWarning('It is not possible to add new triggers while the device is running. Stop the device and perform all setup before starting.')
        else:
            self.__triggers.append(trigger)
            trigger._device = self

    def remove_trigger(self, trigger):
        # TODO: Documentation
        if self.__main_thread.is_alive():
            # TODO: Custom warning class
            raise UserWarning('It is not possible to remove triggers while the device is running. Stop the device and perform all setup before starting.')
        else:
            self.__triggers.remove(trigger)
            trigger._device = None

    def add_generator(self, generator):
        if self.__main_thread.is_alive():
            raise UserWarning('It is not possible to add new generators while the device is running. Stop the device and perform all setup before starting.')
        else:
            self.__generators.append(generator)
            generator._device = self

    def remove_generator(self, generator):
        if self.__main_thread.is_alive():
            raise UserWarning('It is not possible to add new generators while the device is running. Stop the device and perform all setup before starting.')
        else:
            self.__generators.remove(generator)
            generator._device = None

    def __reset(self):
        self.__main_stop_event.clear()
        self.__trigger_stop_event.clear()
        self.__q_stop_event.clear()
        self._hardware_reset()
        self._hardware_stop_event.clear()
        for trigger in self.__triggers:
            trigger.reset()
        for generator in self.__generators:
            generator.reset()
        self.input_active.clear()
        self.output_active.clear()

    def _Device__main_target(self):
        # The explicit naming of this method is needed on windows for some stange reason.
        # If we rely on the automatic name wrangling for the process target, it will not be found in device subclasses.
        self._hardware_input_Q = queue.Queue()
        self._hardware_output_Q = queue.Queue(maxsize=25)
        self._hardware_stop_event = threading.Event()
        self.__triggered_q = queue.Queue()
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
        self._hardware_stop()
        hardware_thread.join()
        self.__trigger_stop_event.set()
        trigger_thread.join()
        self.__q_stop_event.set()
        q_thread.join()
        self.__reset()
        # TODO: We will not finish the process if there are items still in the Q,
        # the question is what we want to do about it?

    def __trigger_target(self):
        # TODO: Get buffer size depending on pre-trigger value
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
            scaled_frame = self.input_scaling(this_frame)
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
            scaled_frame = self.input_scaling(this_frame)
            for trig in self.__triggers:
                trig(scaled_frame)
            data_buffer.append(this_frame)
            if self.input_active.is_set():
                while len(data_buffer) > 0:
                    self.__triggered_q.put(data_buffer.popleft())

    def __q_target(self):
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


class Channel:
    @classmethod
    def from_json(cls, json_dict):
        return cls(**json.loads(json_dict))

    def __init__(self, index, chtype, label=None, calibration=None, unit=None):
        self.index = index
        self.chtype = chtype
        self.label = label
        self.calibration = calibration
        self.unit = unit

    def __eq__(self, other):
        try:
            chtype_eq = self.chtype == other.chtype
        except AttributeError:
            chtype_eq = True
        return self.index == other and chtype_eq

    def __int__(self):
        return self.index

    def __repr__(self):
        return self.__class__.__name__ + '(' + ', '.join(['{}={}'.format(key, value) for key, value in self.__dict__.items()]) + ')'

    def __str__(self):
        label_str = '' if self.label is None else ' "{}"'.format(self.label)
        calib_str = '' if self.calibration is None else ' ({:.4g} {})'.format(self.calibration, self.unit)
        return '{chtype} channel {index}{label}{calib}'.format(chtype=self.chtype, index=self.index, label=label_str, calib=calib_str).capitalize()

    def to_json(self):
        return json.dumps(self.__dict__)


class Trigger:
    # TODO: Documentation
    def __init__(self, action=None, false_action=None, auto_deactivate=True):
        # self.active = multiprocessing.Event()
        self.active = threading.Event()
        self.active.set()

        self.actions = []
        self.auto_deactivate = auto_deactivate
        if action is not None:
            try:
                self.actions.extend(action)
            except TypeError:
                self.actions.append(action)

        self.false_actions = []
        if false_action is not None:
            try:
                self.false_actions.extend(false_action)
            except TypeError:
                self.false_actions.append(false_action)
        self._device = None
        self.use_calibrations = False

    def __call__(self, frame):
        # We need to perform the test event if the triggering is disabled
        # Some triggers (RMSTrigger) needs to update their state continiously to work as intended
        # If e.g. RMSTrigger cannot update the level with the triggering disabled, it will always
        # start form zero
        test = self.test(frame * self.calibrations)
        if self.active.is_set():
            # logger.debug('Testing in {}'.format(self.__class__.__name__))
            if test:
                [action() for action in self.actions]
            else:
                [action() for action in self.false_actions]

    def test(self, frame):
        raise NotImplementedError('Required method `test` is not implemented in {}'.format(self.__class__.__name__))

    def reset(self):
        self.active.set()

    def setup(self):
        if self.use_calibrations:
            self.calibrations = self._device.calibrations
        else:
            self.calibrations = np.ones(len(self._device.inputs))

    @property
    def auto_deactivate(self):
        return self.active.clear in self.actions

    @auto_deactivate.setter
    def auto_deactivate(self, value):
        if value and not self.auto_deactivate:
            self.actions.insert(0, self.active.clear)
        elif self.auto_deactivate and not value:
            self.actions.remove(self.active.clear)


class Generator:
    def __init__(self):
        self._device = None

    def __call__(self):
        return np.atleast_2d(self.frame())

    def frame(self):
        raise NotImplementedError('Required method `frame` is not implemented in {}'.format(self.__class__.__name__))

    def reset(self):
        pass

    def setup(self):
        pass


class GeneratorStop(Exception):
        pass
