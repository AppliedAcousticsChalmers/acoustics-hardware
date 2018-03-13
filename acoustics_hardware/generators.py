import numpy as np
from numpy.fft import rfft as fft, irfft as ifft
from scipy.signal import waveforms, max_len_seq
import queue
from . import core, utils


class QGenerator(core.Generator):
    def __init__(self):
        core.Generator.__init__(self)
        self.Q = queue.Queue()
        self.buffer = None

    def frame(self):
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
        core.Generator.reset(self)
        utils.flush_Q(self.Q)


class ArbitrarySignalGenerator(core.Generator):
    def __init__(self, repetitions=np.inf, **kwargs):
        core.Generator.__init__(self)
        self.repetitions = repetitions  # Default to continious output
        self.kwargs = kwargs
        self.reset()

    def frame(self):
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
        core.Generator.reset(self)
        self.signal = None
        self.idx = 0
        self.repetitions_done = 0

    def setup(self):
        """
        Implements signal creation. Create one cycle of the signal and store it
        in `self.signal`. Access the underlying device as `self._device`, which has
        important properties, e.g. samplerate `fs`.
        """
        core.Generator.setup(self)
        if 'signal' in self.kwargs:
            self.signal = self.kwargs['signal']


class SweepGenerator(ArbitrarySignalGenerator):
    def __init__(self, start_frequency, stop_frequency, duration, method='logarithmic', bidirectional=False, **kwargs):
        ArbitrarySignalGenerator.__init__(self, **kwargs)
        self.start_frequency = start_frequency
        self.stop_frequency = stop_frequency
        self.duration = duration
        self.method = method
        self.bidirectional = bidirectional

    def setup(self):
        ArbitrarySignalGenerator.setup(self)
        time_vector = np.arange(round(self.duration * self._device.fs)) / self._device.fs
        self.signal = waveforms.chirp(time_vector, self.start_frequency, self.duration, self.stop_frequency, method=self.method, phi=90)
        if self.bidirectional:
            self.signal = np.concatenate([self.signal, self.signal[::-1]])
            self.repetitions /= 2


class MaximumLengthSequenceGenerator(ArbitrarySignalGenerator):
    def __init__(self, order, **kwargs):
        ArbitrarySignalGenerator.__init__(self, **kwargs)
        self.order = order

    def setup(self):
        ArbitrarySignalGenerator.setup(self)
        self.sequence, state = max_len_seq(self.order)
        self.signal = 1 - 2 * self.sequence


class FunctionGenerator(core.Generator):
    _functions = {
        'sin': waveforms.sin,
        'saw': waveforms.sawtooth,
        'squ': waveforms.square
    }

    def __init__(self, frequency, amplitude=1, repetitions=np.inf, shape='sine', phase_offset=0, **kwargs):
        core.Generator.__init__(self)
        self.repetitions = repetitions  # Default to continious output
        self.kwargs = kwargs
        self.frequency = frequency
        self.amplitude = amplitude
        self.shape = shape
        self.phase_offset = phase_offset

    def frame(self):
        if self.repetitions_done >= self.repetitions:
            raise core.GeneratorStop('Finite number of repetitions reached')
        frame = self._function(self._phase_array + self._phase, **self.kwargs)
        self._phase += self._phase_per_frame
        if self.repetitions_done >= self.repetitions:
            surplus_reps = self.repetitions_done - self.repetitions
            surplus_samps = round(surplus_reps * self._device.fs / self.frequency)
            frame[-surplus_samps:] = 0
        return self.amplitude * frame

    def setup(self):
        core.Generator.setup(self)
        self.reset()
        taps = np.arange(self._device.framesize)
        self._phase_array = 2 * np.pi * taps * self.frequency / self._device.fs
        self._phase_per_frame = 2 * np.pi * self.frequency / self._device.fs * self._device.framesize

    def reset(self):
        core.Generator.reset(self)
        self._phase = self.phase_offset

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, val):
        if val.lower()[:3] in self._functions:
            self._shape = val.lower()
            self._function = self._functions[val.lower()[:3]]
        else:
            raise KeyError('Unknown function shape `{}`'.format(val))

    @property
    def repetitions_done(self):
        return self._phase / (2 * np.pi)


class NoiseGenerator(core.Generator):
    _color_slopes = {
        'purple': -2,
        'blue': -1,
        'white': 0,
        'pink': 1,
        'brown': 2
    }

    def __init__(self, color='white', method='autoregressive', **kwargs):
        core.Generator.__init__(self)
        self.color = color
        self.method = method
        self.method_args = kwargs

    def _fft_noise(self):
        normal = np.random.normal(size=self._device.framesize)
        shaped = ifft(self._spectral_coefficients * fft(normal))
        # TODO: Normalization, variable amplitude, soft clipping?
        return shaped

    def _fft_setup(self):
        bins = np.arange(self._device.framesize // 2 + 1)  # We are using single sided spectrum
        bins[0] = 1  # Do not modify DC bin
        self._spectral_coefficients = bins.astype('double')**(-self.power / 2)

    def _fft_reset(self):
        del self._spectral_coefficients

    def _ar_noise(self):
        normal = np.random.normal(size=self._device.framesize)
        shaped = np.zeros(shape=self._device.framesize)
        for idx in range(self._device.framesize):
            shaped[idx] = normal[idx] - (self._ar_coefficients * self._ar_buffer).sum()
            self._ar_buffer = np.roll(self._ar_buffer, 1)
            self._ar_buffer[0] = shaped[idx]
        return shaped

    def _ar_setup(self):
        order = self.method_args.get('order', 63)
        self._ar_buffer = np.zeros(order - 1)
        coefficients = np.zeros(order)
        coefficients[0] = 1
        for k in range(1, order):
            coefficients[k] = (k - 1 - self.power / 2) * coefficients[k - 1] / k
        self._ar_coefficients = coefficients[1:]

    def _ar_reset(self):
        del self._ar_buffer
        del self._ar_coefficients

    _call_methods = {
        'fft': _fft_noise,
        'autoregressive': _ar_noise
    }

    _setup_methods = {
        'fft': _fft_setup,
        'autoregressive': _ar_setup
    }

    _reset_methods = {
        'fft': _fft_reset,
        'autoregressive': _ar_reset
    }

    def frame(self):
        return self._call_methods[self.method](self)

    def setup(self):
        core.Generator.setup(self)
        self._setup_methods[self.method](self)

    def reset(self):
        core.Generator.reset(self)
        self._reset_methods[self.method](self)

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, val):
        try:
            self.power = self._color_slopes[val.lower()]
        except KeyError:
            raise KeyError('Unknown noise color `{}`'.format(val))
        except AttributeError:
            self.power = val
            self._color = 'custom'
        else:
            self._color = val.lower()

    @property
    def method(self):
        return self._method

    @method.setter
    def method(self, val):
        if val.lower() in self._call_methods:
            self._method = val.lower()
        else:
            raise KeyError('Unknown generation method `{}`'.format(val))
