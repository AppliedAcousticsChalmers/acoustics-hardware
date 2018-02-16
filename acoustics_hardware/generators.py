import numpy as np
from numpy.fft import rfft as fft, irfft as ifft
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

    def __call__(self):
        return self._call_methods[self.method](self)

    def setup(self):
        self._setup_methods[self.method](self)

    def reset(self):
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
