import numpy as np

from . import _core, signal_tools


class _Generator(_core.SamplerateFollower):
    def process(self, frame):
        return np.atleast_2d(frame)


class SignalGenerator(_Generator):
    def __init__(self, signal, repetitions=1, **kwargs):
        super().__init__(**kwargs)
        self.signal = signal

        if repetitions == -1 or (isinstance(repetitions, str) and repetitions.lower()[:3] == 'inf'):
            repetitions = np.inf
        self.repetitions = repetitions
        self.reset()

    def reset(self, **kwargs):
        super().reset(**kwargs)
        self._repetitions_done = 0
        self._sample_index = 0

    def process(self, framesize):
        if self._repetitions_done >= self.repetitions:
            raise _core.PipelineStop()

        start_idx = self._sample_index
        stop_idx = self._sample_index + framesize
        signal_length = self.signal.shape[-1]

        if stop_idx <= signal_length:
            self._sample_index += framesize
            frame = self.signal[..., start_idx:stop_idx]
        else:
            self._repetitions_done += 1
            first_part = self.signal[..., start_idx:]
            if self._repetitions_done < self.repetitions:
                # We should keep repeating the signal
                self._sample_index = stop_idx % signal_length
                second_part = self.signal[..., :self._sample_index]
                frame = np.concatenate([first_part, second_part], axis=-1)
            else:
                frame = signal_tools.extend_signals(first_part, length=framesize)
        return super().process(frame)

