from . import _core, signal_tools
import numpy as np
import scipy.signal


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


class SweepGenerator(SignalGenerator):
    def __init__(
        self,
        lower_frequency, upper_frequency,
        sweep_length, method='logarithmic',
        **kwargs
    ):
        super().__init__(**kwargs)
        self.lower_frequency = lower_frequency
        self.upper_frequency = upper_frequency
        self.sweep_length = sweep_length
        self.method = method

    def setup(self, **kwargs):
        super().setup(**kwargs)
        time_vector = np.arange(round(self.sweep_length * self.samplerate)) / self.samplerate
        signal = scipy.signal.chirp(t=time_vector, f0=self.lower_frequency, t1=self.sweep_length, f1=self.upper_frequency, method=self.method, phi=-90)
        last_crossing = -1
        while np.sign(signal[last_crossing - 1]) == np.sign(signal[-1]):
            last_crossing -= 1
        signal[last_crossing:] = 0
        self.signal = signal


class MaximumLengthGenerator(SignalGenerator):
    def __init__(self, order, **kwargs):
        super().__init__(**kwargs)
        self.order = order

    def setup(self, **kwargs):
        super().setup(**kwargs)
        self.sequence, _ = scipy.signal.max_len_seq(self.order)
        self.signal = (1 - 2 * self.sequence).astype('float64')


class ToneGenerator(_Generator):
    _functions = {
        'sin': np.sin,
        'saw': scipy.signal.sawtooth,
        'squ': scipy.signal.square
    }

    def __init__(
        self,
        frequency,
        periods=None, duration=None,
        phase_offset=0,
        shape='sine',
        shape_kwargs=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.frequency = frequency
        self.shape = shape
        self.shape_kwargs = shape_kwargs if shape_kwargs is not None else {}
        self.phase_offset = phase_offset

        if periods is None and duration is None:
            self.periods = np.inf
        elif periods is not None:
            self.periods = periods
        elif duration is not None:
            self.periods = duration * self.frequency
        else:
            raise ValueError('Cannot specify both duration and number of repetitions')

        self.reset()

    def reset(self, **kwargs):
        super().reset(**kwargs)
        self._phase = 0

    def process(self, framesize):
        if self._phase >= self.periods * 2 * np.pi:
            raise _core.PipelineStop()

        phase = np.arange(framesize) * (2 * np.pi * self.frequency / self.samplerate)
        signal = self._function(self._phase + self.phase_offset + phase, **self.shape_kwargs)
        self._phase += 2 * np.pi * framesize * self.frequency / self.samplerate
        if self._phase > self.periods * 2 * np.pi:
            phase_to_mute = self._phase - self.periods * 2 * np.pi
            samples_to_mute = np.math.floor(phase_to_mute / (2 * np.pi * self.frequency) * self.samplerate)
            signal[-samples_to_mute:] = 0
        return signal

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, value):
        if value[:3].lower() in self._functions:
            self._shape = value
            self._function = self._functions[value[:3].lower()]
        else:
            raise ValueError(f'Unknown tone shape {value}')
