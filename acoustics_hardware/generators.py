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
            frame = self.Q.get(timeout=self._device._generator_timeout)
            gen_frame.append(frame[..., :samples_left])
            samples_left -= frame.shape[-1]
        if samples_left < 0:
            self.buffer = frame[..., samples_left:]
        else:
            self.buffer = None
        return np.concatenate(gen_frame, axis=-1)

    def reset(self):
        utils.flush_Q(self.Q)
