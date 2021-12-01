import numpy as np
import scipy.signal


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
        super().__init__(**kwargs)
        self.time_constant = time_constant

        self.channel = channel
        self.reset()

    def process(self, frame):
        # Squared input level is tracked in order for the RMS trigger to work properly.
        input_levels = frame[self.channel]**2
        output_levels, self._buffer = scipy.signal.lfilter([self._digital_constant], [1, self._digital_constant - 1], input_levels, zi=self._buffer)
        return output_levels**0.5

    def reset(self):
        super().reset()
        self._buffer = np.atleast_1d(0)

    @property
    def time_constant(self):
        return -1 / (np.log(1 - self._digital_constant) * self.device.fs)

    @time_constant.setter
    def time_constant(self, val):
        self._digital_constant = 1 - np.exp(-1 / (val * self.device.fs))

    @property
    def current_level(self):
        return self._buffer[0]**0.5
