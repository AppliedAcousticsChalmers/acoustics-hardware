import numpy as np
from scipy.signal import lfilter
import queue


def flush_Q(q):
    while True:
        try:
            q.get(timeout=0.1)
        except queue.Empty:
            break


class LevelDetector:
    def __init__(self, channel, fs, time_constant=50e-3):
        self.fs = fs  # TODO: Warning if the sampling frequency is no set? Or just wait until we start and crash everything.
        self.time_constant = time_constant

        # TODO: Multichannel level detector?
        self.channel = channel
        self.reset()

    def __call__(self, block):
        # TODO: Enable custom mappings?
        # Squared input level is tracked in order for the RMS trigger to work properly.
        input_levels = block[self.channel]**2
        output_levels, self._buffer = lfilter([self._digital_constant], [1, self._digital_constant - 1], input_levels, zi=self._buffer)
        return output_levels**0.5

    def reset(self):
        self._buffer = np.atleast_1d(0)

    @property
    def time_constant(self):
        return -1 / (np.log(1 - self._digital_constant) * self.fs)

    @time_constant.setter
    def time_constant(self, val):
        self._digital_constant = 1 - np.exp(-1 / (val * self.fs))

    @property
    def current_level(self):
        return self._buffer**0.5


class LevelDetectorAttackRelease:
    def __init__(self, *, channel, fs, time_constant=None, attack_time=None, release_time=None):
        self.fs = fs  # TODO: Warning if the sampling frequency is no set? Or just wait until we start and crash everything.
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

        # TODO: Multichannel level detector?
        self.channel = channel
        self.current_level = 0

    # TODO: If the loop is not fast enough there are a few options.
    # If we drop the possibility for separate attack and release times, the level detector is
    # just a IIR filter, so we could use scipy.signal.lfilter([1, (1-alpha)], alpha, input_levels)
    # It should also be possible (I think) to write the level detector using Faust, and just drop in the
    # wrapped Faust code here. If I manage to get that working it could also be used for all other crazy
    # filters we might want to use.
    def __call__(self, block):
        # TODO: Make the function configurable, e.g. abs() or **2 etc.
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
            # TODO: Warn?
        self.attack_time = value[0] * 2.2
        self.release_time = value[1] * 2.2
