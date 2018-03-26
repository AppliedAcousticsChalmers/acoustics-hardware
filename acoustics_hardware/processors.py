import numpy as np
from scipy.signal import lfilter


class Processor:
    """Base class for processors

    A processor is an object that manipulates the data in some way.
    """
    def __init__(self, device=None, **kwargs):
        self.device = device
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __call__(self, frame):
        return self.process(frame)

    def process(self, frame):
        """Processes a single fraame of input.

        The input frame might be the rame object as the read frame, so a
        processor should not manitulate the data in place.

        Arguments:
            frame (`numpy.ndarray`): ``(n_ch, n_samp)`` shape input frame.
        """

    def setup(self):
        pass

    def reset(self):
        pass

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device):
        self._device = device


class LevelDetector(Processor):
    """Single channel level detector.

    A level detector that tracks the root-mean-square level in a signal.
    The level is tracked with an exponentially decaying time average,
    implemented as a low-passed squared amplitude.

    Arguments:
        channel (`int`): The index of the channel to track.
        time_constant (`float`): Time constant for the exponential decay in
            seconds, default 50 ms.
        **kwargs: Extra arguments will be passed to `Processor`.
    """
    def __init__(self, channel, time_constant=50e-3, **kwargs):
        Processor.__init__(self, **kwargs)
        self.time_constant = time_constant

        self.channel = channel
        self.reset()

    def process(self, frame):
        # Squared input level is tracked in order for the RMS trigger to work properly.
        input_levels = frame[self.channel]**2
        output_levels, self._buffer = lfilter([self._digital_constant], [1, self._digital_constant - 1], input_levels, zi=self._buffer)
        return output_levels**0.5

    def reset(self):
        self._buffer = np.atleast_1d(0)

    @property
    def time_constant(self):
        return -1 / (np.log(1 - self._digital_constant) * self.device.fs)

    @time_constant.setter
    def time_constant(self, val):
        self._digital_constant = 1 - np.exp(-1 / (val * self.device.fs))

    @property
    def current_level(self):
        return self._buffer**0.5


class LevelDetectorAttackRelease:
    def __init__(self, *, channel, fs, time_constant=None, attack_time=None, release_time=None):
        self.fs = fs
        if attack_time is not None and release_time is not None:
            # separate attack and release times specified, used them
            # Attack and release time are defined as the time it takes for a step
            # to rise from 10% to 90%.
            self.attack_time = attack_time
            self.release_time = release_time
        elif time_constant is not None:
            # Single number or two numbers
            self.time_constant = time_constant
        else:
            self.time_constant = 50e-3  # Default value of 50 ms

        self.channel = channel
        self.current_level = 0

    def __call__(self, block):
        """
        Todo:
            This is too slow. If this style of detector is kept it must be improved
            with better processing. Possibilities are cython, numbe, or faust.
        """
        input_levels = np.abs(block[self.channel])
        output_levels = np.empty_like(input_levels)
        level = self.current_level  # Get the level after the privious block
        for idx in range(len(input_levels)):
            if input_levels[idx] > level:
                level = self._attack_constant * input_levels[idx] + (1 - self._attack_constant) * level
            else:
                level = self._release_constant * input_levels[idx] + (1 - self._release_constant) * level
            output_levels[idx] = level
        self.current_level = level  # Save the level to use in the next block
        return output_levels

    @property
    def attack_time(self):
        return -2.2 / self.fs / np.log(1 - self._attack_constant)

    @attack_time.setter
    def attack_time(self, value):
        self._attack_constant = 1 - np.exp(-2.2 / (value * self.fs))

    @property
    def release_time(self):
        return -2.2 / self.fs / np.log(1 - self._release_constant)

    @release_time.setter
    def release_time(self, value):
        self._release_constant = 1 - np.exp(-2.2 / (value * self.fs))

    @property
    def time_constant(self):
        values = (self.attack_time, self.release_time)
        if (values[1] - values[0]) * 2 / (values[1] + values[0]) < 0.01:
            # The relative difference is less than 1%, retuen just a single value
            return values[0]
        else:
            return values

    @time_constant.setter
    def time_constant(self, value):
        try:
            len(value)
        except TypeError:
            value = (value, value)

        # if len(value) is not 2:
        self.attack_time = value[0] * 2.2
        self.release_time = value[1] * 2.2
