from . import _core, signal_tools
import numpy as np


class _Generator(_core.SamplerateFollower):
    def process(self, frame):
        return np.atleast_2d(frame)


class SignalGenerator(_Generator):
    def __init__(
        self, signal=None, repetitions=1,
        fade_in=None, fade_out=None,
        pre_pad=None, post_pad=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.fade_in = fade_in
        self.fade_out = fade_out
        self.pre_pad = pre_pad
        self.post_pad = post_pad

        if signal is not None:
            self.signal = signal

        if repetitions == -1 or (isinstance(repetitions, str) and repetitions.lower()[:3] == 'inf'):
            repetitions = np.inf
        self.repetitions = repetitions
        self.reset()

    def reset(self, **kwargs):
        super().reset(**kwargs)
        self._repetitions_done = 0
        self._sample_index = 0

    @property
    def signal(self):
        return self._signal

    @signal.setter
    def signal(self, signal):
        signal = signal_tools.fade_signals(signal, fade_in=self.fade_in, fade_out=self.fade_out, samplerate=self.samplerate, inplace=False)
        signal = signal_tools.pad_signals(signal, pre_pad=self.pre_pad, post_pad=self.post_pad, samplerate=self.samplerate)
        self._signal = signal

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
                if first_part.size == 0:
                    raise _core.PipelineStop()
                frame = signal_tools.extend_signals(first_part, length=framesize)
        return super().process(frame)

