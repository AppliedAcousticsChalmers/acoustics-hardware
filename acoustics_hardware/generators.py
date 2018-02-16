import numpy as np
import queue
from . import core, utils


class QGenerator(core.Generator):
    def __init__(self):
        core.Generator.__init__(self)
        self.Q = queue.Queue()
        self.buffer = None

    def __call__(self):
        gen_frame = []
        samples_left = self._device.framesize
        if self.buffer is not None:
            gen_frame.append(self.buffer[..., :samples_left])
            samples_left -= gen_frame[-1].shape[-1]
        while samples_left > 0:
            try:
                frame = self.Q.get(timeout=self._device._generator_timeout)
            except queue.Empty:
                raise core.GeneratorStop('Input Q is empty')
            gen_frame.append(frame[..., :samples_left])
            samples_left -= frame.shape[-1]
        if samples_left < 0:
            self.buffer = frame[..., samples_left:]
        else:
            self.buffer = None
        return np.concatenate(gen_frame, axis=-1)

    def reset(self):
        utils.flush_Q(self.Q)


class SignalGenerator(core.Generator):
    def __init__(self, repetitions=np.inf):
        core.Generator.__init__(self)
        self.signal = None
        self.repetitions = repetitions  # Default to continious output
        self.reset()

    def __call__(self):
        if self.repetitions_done >= self.repetitions:
            raise core.GeneratorStop('Finite number of repetitions reached')
        samples_left = self._device.framesize
        gen_frame = []
        while samples_left > 0:
            gen_frame.append(self.signal[..., self.idx:self.idx + samples_left])
            samps_app = gen_frame[-1].shape[-1]
            samples_left -= samps_app
            self.idx += samps_app
            if self.idx >= self.signal.shape[-1]:
                self.repetitions_done += 1
                self.idx = 0
                if self.repetitions_done >= self.repetitions:
                    gen_frame.append(np.zeros(self.signal.shape[:-1] + (samples_left,)))
                    samples_left = 0
        return np.concatenate(gen_frame, axis=-1)

    def reset(self):
        self.idx = 0
        self.repetitions_done = 0

    def setup(self):
        """
        Implements signal creation. Create one cycle of the signal and store it
        in `self.signal`. Access the underlying device as `self._device`, which has
        important properties, e.g. samplerate `fs`.
        """
        raise NotImplementedError('Required method `setup` not implemented in {}'.format(self.__class__.__name__))


class SineGenerator(SignalGenerator):
    def __init__(self, frequency=None, amplitude=1, **kwargs):
        SignalGenerator.__init__(self, **kwargs)
        self.frequency = frequency
        self.amplitude = amplitude

    def setup(self):
        samps = round(self._device.fs / self.frequency)
        self.frequency = self._device.fs / samps
        self.signal = self.amplitude * np.sin(np.arange(samps) / samps * 2 * np.pi)
