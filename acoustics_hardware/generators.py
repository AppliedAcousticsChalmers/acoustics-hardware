from . import _core, signal_tools
import numpy as np
import scipy.signal


class _Generator(_core.SamplerateFollower):
    def process(self, packet):
        if not isinstance(packet, _core.FrameRequest):
            return packet
        frame = self.generate(packet.framesize)
        if isinstance(frame, _core.Frame):
            frame.frame = np.atleast_2d(frame.frame)
        elif not isinstance(frame, _core.Packet):
            frame = _core.Frame(np.atleast_2d(frame))
        return frame

    def generate(self, framesize):
        return np.zeros(framesize)


class SignalGenerator(_Generator):
    def __init__(
        self, signal=None, repetitions=1,
        amplitude=1,
        fade_in=None, fade_out=None,
        pre_pad=None, post_pad=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.amplitude = amplitude
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
        try:
            return self._signal
        except AttributeError:
            pass
        self.setup()
        return self._signal

    @signal.setter
    def signal(self, signal):
        signal = signal_tools.fade_signals(signal, fade_in=self.fade_in, fade_out=self.fade_out, samplerate=self.samplerate, inplace=False)
        signal = signal_tools.pad_signals(signal, pre_pad=self.pre_pad, post_pad=self.post_pad, samplerate=self.samplerate)
        self._signal = signal * self.amplitude

    def generate(self, framesize):
        if self._repetitions_done >= self.repetitions:
            return _core.LastFrame(np.zeros(framesize), 0)

        start_idx = self._sample_index
        stop_idx = self._sample_index + framesize
        signal_length = self.signal.shape[-1]

        if stop_idx <= signal_length:
            frame = self.signal[..., start_idx:stop_idx]
        else:
            num_repeats = np.math.ceil(stop_idx / signal_length)
            repeated_signal = np.tile(self.signal, num_repeats)
            frame = repeated_signal[..., start_idx:stop_idx]

        repeats, self._sample_index = divmod(stop_idx, signal_length)
        self._repetitions_done += repeats
        if self._repetitions_done >= self.repetitions:
            if self._sample_index > 0:
                frame[..., -self._sample_index:] = 0
            frame = _core.LastFrame(frame, framesize - self._sample_index)
        else:
            frame = _core.Frame(frame)
        return frame

    def once(self):
        if self._sample_index != 0 and self._repetitions_done != 0:
            raise RuntimeError('Cannot use method `once` on generator which is running in a streamed pipeline!')
        framesize = self.repetitions * self.signal.shape[-1]

        frame = self.request(_core.FrameRequest(framesize))
        self._sample_index = self._repetitions_done = 0
        return self.push(frame)


class SweepGenerator(SignalGenerator):
    def __init__(
        self,
        lower_frequency, upper_frequency,
        sweep_length, method='logarithmic',
        amplitude_slope=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.lower_frequency = lower_frequency
        self.upper_frequency = upper_frequency
        self.sweep_length = sweep_length
        self.amplitude_slope = amplitude_slope
        self.method = method

    def setup(self, **kwargs):
        super().setup(**kwargs)
        time_vector = np.arange(round(self.sweep_length * self.samplerate)) / self.samplerate
        signal = scipy.signal.chirp(t=time_vector, f0=self.lower_frequency, t1=self.sweep_length, f1=self.upper_frequency, method=self.method, phi=-90)
        last_crossing = -1
        while np.sign(signal[last_crossing - 1]) == np.sign(signal[-1]):
            last_crossing -= 1
        signal[last_crossing:] = 0
        if self.amplitude_slope is not None:
            end_value = 10**(6 * self.amplitude_slope / 20 * np.log2(self.upper_frequency / self.lower_frequency))
            if self.method == 'logarithmic':
                slope = np.geomspace(1, end_value, signal.size)
            elif self.method == 'linear':
                slope = np.linspace(1, end_value, signal.size)
            else:
                raise ValueError(f'Cannot use amplitude compensation with sweep method {self.method}')
            slope /= max(slope)
        else:
            slope = 1
        self.reference = signal.copy()
        self.signal = signal * slope


class MaximumLengthGenerator(SignalGenerator):
    def __init__(self, order=None, sequence_time=None, **kwargs):
        super().__init__(**kwargs)
        if None not in (order, sequence_time):
            raise ValueError('Cannot give both MLS order and MLS time')
        elif sequence_time is not None:
            self._sequence_time = sequence_time
        elif order is not None:
            self.order = order

    @property
    def sequence_time(self):
        try:
            return self._sequence_time
        except AttributeError:
            pass
        try:
            return (2**self.order - 1) / self.samplerate
        except AttributeError:
            raise AttributeError('Cannot use MLS sequence time without a defined samplerate')

    @sequence_time.setter
    def sequence_time(self, value):
        try:
            n_samples = self.samplerate * value
        except AttributeError:
            self._sequence_time = value
            del self._order
        else:
            self.order = np.math.ceil(np.log2(n_samples + 1))

    @property
    def order(self):
        try:
            return self._order
        except AttributeError:
            pass
        try:
            n_samples = self.samplerate * self._sequence_time
        except AttributeError:
            raise AttributeError('Cannot use MLS sequence time without a defined samplerate')
        return np.math.ceil(np.log2(n_samples + 1))

    @order.setter
    def order(self, value):
        if not isinstance(value, int):
            raise TypeError(f"Cannot set order to non-integer value {value}")
        self._order = value
        try:
            del self._sequence_time
        except AttributeError:
            pass

    def setup(self, **kwargs):
        super().setup(**kwargs)
        self.reference, _ = scipy.signal.max_len_seq(self.order)
        self.signal = (1 - 2 * self.reference).astype('float64')


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

    def generate(self, framesize):
        if self._phase >= self.periods * 2 * np.pi:
            return _core.LastFrame(np.zeros(framesize), 0)

        phase = np.arange(framesize) * (2 * np.pi * self.frequency / self.samplerate)
        signal = self._function(self._phase + self.phase_offset + phase, **self.shape_kwargs)
        self._phase += 2 * np.pi * framesize * self.frequency / self.samplerate
        if self._phase > self.periods * 2 * np.pi:
            phase_to_mute = self._phase - self.periods * 2 * np.pi
            samples_to_mute = np.math.floor(phase_to_mute / (2 * np.pi * self.frequency) * self.samplerate)
            signal[-samples_to_mute:] = 0
            frame = _core.LastFrame(signal, framesize - samples_to_mute)
        else:
            frame = _core.Frame(signal)
        return frame

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
