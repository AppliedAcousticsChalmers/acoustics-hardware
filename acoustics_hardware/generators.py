import numpy as np
from numpy.fft import rfft as fft, irfft as ifft
import scipy.signal
import queue
import warnings
from . import utils


class Generator:
    """Base class for generator implementations.

    A `Generator` is an object that creates data for output channels in a
    Device. Refer to specific generators for more details.

    Attributes:
        amplitude (`float`): The amplitude scale of the generator.
    """
    def __init__(self, device=None, amplitude=1, **kwargs):
        self.device = device
        self.amplitude = amplitude
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __call__(self):
        """Manages frame creation"""
        return self.amplitude * np.atleast_2d(self.frame())

    def frame(self):
        """Generates a frame of output.

        The generated frame must match the device framesize.
        If the generator creates multiple channels, it should have the shape
        ``(n_ch, framesize)``, otherwise 1d arrays are sufficient.

        Returns:
            `numpy.ndarray`: Generated frame.
        """
        raise NotImplementedError('Required method `frame` is not implemented in {}'.format(self.__class__.__name__))

    def reset(self):
        """Resets the generator."""
        pass

    def setup(self):
        """Configures the generator state."""
        pass

    @property
    def device(self):
        try:
            return self._device
        except AttributeError:
            return None

    @device.setter
    def device(self, dev):
        if self.device is not None:
            # Unregister from the previous device
            if self.device.initialized:
                self.reset()
            # noinspection PyProtectedMember
            self.device._Device__generators.remove(self)
        self._device = dev
        if self.device is not None:
            # Register to the new device
            # noinspection PyProtectedMember
            self.device._Device__generators.append(self)
            if self.device.initialized:
                self.setup()


class GeneratorStop(Exception):
    """Raised by Generators.

    This exception indicates that the generator have reached some stopping
    criteria, e.g. end of file. Should be caught by the Device to stop output.
    """
    pass


class QGenerator(Generator):
    """Generator using `queue.Queue`.

    Implementation of a `Generator` using a queue.
    Takes data from an input queue and generates frames with the correct
    framesize. The input queue must be filled fast enough otherwise the
    device output is cancelled.

    Attributes:
        Q (`~queue.Queue`): The queue from where data is extracted.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.Q = queue.Queue()
        self.buffer = None

    def frame(self):
        gen_frame = []
        samples_left = self.device.framesize
        if self.buffer is not None:
            gen_frame.append(self.buffer[..., :samples_left])
            samples_left -= gen_frame[-1].shape[-1]
        while samples_left > 0:
            try:
                # noinspection PyProtectedMember
                frame = self.Q.get(timeout=self.device._generator_timeout)
            except queue.Empty:
                raise GeneratorStop('Input Q is empty')
            gen_frame.append(frame[..., :samples_left])
            samples_left -= frame.shape[-1]
        if samples_left < 0:
            self.buffer = frame[..., samples_left:]
        else:
            self.buffer = None
        return np.concatenate(gen_frame, axis=-1)

    def reset(self):
        """Clears the input queue."""
        super().reset()
        utils.flush_Q(self.Q)


class ArbitrarySignalGenerator(Generator):
    """Repeated generation of arbitrary signals.

    Implementation of `Generator` for arbitrary signals.

    Arguments:
        repetitions (`float`): The number of cycles to output before stopping,
        default `np.inf`. **kwargs: Will be saved as ``kwargs`` and
        accessible in `setup`.
    Keyword Arguments:
        signal (`numpy.ndarray`): One cycle of the signal to output.
    """
    def __init__(self, repetitions=np.inf, **kwargs):
        super().__init__(**kwargs)
        self.repetitions = repetitions  # Default to continuous output
        self.signal = None
        self.idx = None
        self.repetitions_done = None
        self.reset()

    def frame(self):
        if self.repetitions_done >= self.repetitions:
            raise GeneratorStop('Finite number of repetitions reached')
        samples_left = self.device.framesize
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
        super().reset()
        self.idx = 0
        self.repetitions_done = 0

    def setup(self):
        """Configures the signal.

        Create the signal manually and pass it while creating the generator
        as the ``signal`` argument.

        It is possible to inherit `ArbitrarySignalGenerator` and override the
        setup method. Create one cycle of the signal and store it in
        ``self.signal``. Access the underlying device as ``self.device``,
        which has important properties, e.g. sampling rate ``fs``.
        All keyword arguments passed while creating instances are available
        as ``self.key``.

        Note:
            Call `ArbitrarySignalGenerator.setup(self)` from subclasses.
        """
        super().setup()


class SweepGenerator(ArbitrarySignalGenerator):
    """Swept sine generator.

    Arguments:
        start_frequency (`float`): Initial frequency of the sweep, in Hz.
        stop_frequency (`float`): Final frequency of the sweep, in Hz.
        duration (`float`): Duration of a single sweep, in seconds.
        repetitions (`float`, optional): The number of repetitions, default `np.inf`.
        method (`str`, optional): Chooses the type of sweep, see
            `~scipy.signal.chirp`, default ``'logarithmic'``.
        bidirectional (`bool`, optional): If the sweep is bidirectional or not,
            default ``False``.
    See Also:
        `ArbitrarySignalGenerator`, `scipy.signal.chirp`
    """
    def __init__(self, start_frequency, stop_frequency, duration,
                 method='logarithmic', bidirectional=False, **kwargs):
        super().__init__(**kwargs)
        self.start_frequency = start_frequency
        self.stop_frequency = stop_frequency
        self.duration = duration
        self.method = method
        self.bidirectional = bidirectional

    def setup(self):
        super().setup()
        time_vector = (np.arange(round(self.duration * self.device.fs))
                       / self.device.fs)
        self.signal = waveforms.chirp(time_vector, self.start_frequency,
                                      self.duration, self.stop_frequency,
                                      method=self.method, phi=90)
        if self.bidirectional:
            self.signal = np.concatenate([self.signal, self.signal[::-1]])
            self.repetitions /= 2


class MaximumLengthSequenceGenerator(ArbitrarySignalGenerator):
    """Generation of maximum length sequences.

    Arguments:
        order (`int`): The order or the sequence. The total length  is
            ``2**order - 1``.
    See Also:
        `ArbitrarySignalGenerator`, `scipy.signal.max_len_seq`
    """
    def __init__(self, order, **kwargs):
        super().__init__(**kwargs)
        self.order = order
        self.sequence = None

    def setup(self):
        super().setup()
        self.sequence, state = scipy.signal.max_len_seq(self.order)
        self.signal = (1 - 2 * self.sequence).astype('float64')


class FunctionGenerator(Generator):
    """Generates signals from a shape function.

    Implementation of `Generator` for standard functions.

    Arguments:
        frequency (`float`): The frequency of the signal, in Hz.
        repetitions (`float`, optional): The number of repetitions,
            default `np.inf`.
        shape (`str`, optional): Function shape, default ``'sine'``. Currently
            available functions are

                - ``'sine'``: `numpy.sin`
                - ``'sawtooth'``: `scipy.signal.sawtooth`
                - ``'square'``: `scipy.signal.square`

        phase_offset (`float`, optional): Phase offset of the signal in radians,
            default 0.
        shape_kwargs (`dict`): Keyword arguments for shape function.
    """
    _functions = {
        'sin': np.sin,
        'saw': scipy.signal.sawtooth,
        'squ': scipy.signal.square
    }

    def __init__(self, frequency, repetitions=np.inf,
                 shape='sine', phase_offset=0, shape_kwargs=None, **kwargs):
        super().__init__(**kwargs)
        self.repetitions = repetitions  # Default to continuous output
        self.frequency = frequency
        self.shape = shape
        self.phase_offset = phase_offset
        self.shape_kwargs = {} if shape_kwargs is None else shape_kwargs
        self._phase = None
        self._phase_array = None
        self._phase_per_frame = None

    def frame(self):
        if self.repetitions_done >= self.repetitions:
            raise GeneratorStop('Finite number of repetitions reached')
        frame = self._function(self._phase_array + self._phase, **self.shape_kwargs)
        self._phase += self._phase_per_frame
        if self.repetitions_done >= self.repetitions:
            surplus_reps = self.repetitions_done - self.repetitions
            surplus_samps = round(surplus_reps * self.device.fs / self.frequency)
            frame[-surplus_samps:] = 0
        return frame

    def setup(self):
        super().setup()
        self.reset()
        taps = np.arange(self.device.framesize)
        self._phase_array = 2 * np.pi * taps * self.frequency / self.device.fs
        self._phase_per_frame = 2 * np.pi * self.frequency / self.device.fs * self.device.framesize

    def reset(self):
        super().reset()
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


class NoiseGenerator(Generator):
    """Generates colored noise.

    Implementation of `Generator` for random noise signals.

    Arguments:
        color (`str`, optional): The color of the noise. Each color corresponds
            to a inverse frequency power in the noise power density spectrum.
            Default is ``'white'``.

            - ``'purple'``: -2
            - ``'blue'``: -1
            - ``'white'``: 0
            - ``'pink'``: 1
            - ``'brown'``: 2
        method (`str`): The method used to create the noise.
            Currently two methods are implemented, a ``'fft'`` method and
            an ``'autoregressive'`` method. The default is ``'autoregressive'``.
            The autoregressive method is more expensive for small framesizes,
            but gives the same performance regardless of the framesize. The
            fft method have bad low-frequency performance for small framesizes.
        ar_order (`int`): The order for the autoregressive method, default 63.
    References:
        N. J. Kasdin, “Discrete simulation of colored noise and stochastic
        processes and 1/f^α power law noise generation,” Proceedings of the
        IEEE, vol. 83, no. 5, pp. 802–827, May 1995.
        :doi:`10.1109/5.381848`
    Todo:
        Variable amplitudes and maximum amplitudes.

    """
    _color_slopes = {
        'purple': -2,
        'blue': -1,
        'white': 0,
        'pink': 1,
        'brown': 2
    }

    def __init__(self, color='white', method='autoregressive',
                 ar_order=63, **kwargs):
        super().__init__(**kwargs)
        self.method = method
        self.ar_order = ar_order
        self.color = color

    def _fft_noise(self):
        normal = np.random.normal(size=self.device.framesize)
        shaped = ifft(self._spectral_coefficients * fft(normal))
        return shaped

    def _fft_setup(self):
        bins = np.arange(self.device.framesize // 2 + 1)  # We are using single sided spectrum
        bins[0] = 1  # Do not modify DC bin
        self._spectral_coefficients = bins.astype('double')**(-self.power / 2)

    def _fft_reset(self):
        del self._spectral_coefficients

    def _ar_noise(self):
        normal = np.random.normal(size=self.device.framesize)
        shaped = np.zeros(shape=self.device.framesize)
        for idx in range(self.device.framesize):
            shaped[idx] = normal[idx] - (self._ar_coefficients * self._ar_buffer).sum()
            self._ar_buffer = np.roll(self._ar_buffer, 1)
            self._ar_buffer[0] = shaped[idx]
        return shaped

    def _ar_setup(self):
        self._ar_buffer = np.zeros(self.ar_order - 1)
        coefficients = np.zeros(self.ar_order)
        coefficients[0] = 1
        for k in range(1, self.ar_order):
            coefficients[k] = (k - 1 - self.power / 2) * coefficients[k - 1] / k
        self._ar_coefficients = coefficients[1:]

    def _ar_reset(self):
        del self._ar_buffer
        del self._ar_coefficients

    _call_methods = {
        'fft': _fft_noise,
        'autoregressive': _ar_noise,
    }

    _setup_methods = {
        'fft': _fft_setup,
        'autoregressive': _ar_setup,
    }

    _reset_methods = {
        'fft': _fft_reset,
        'autoregressive': _ar_reset,
    }

    def frame(self):
        # noinspection PyArgumentList
        return self._call_methods[self.method](self)

    def setup(self):
        super().setup()
        # noinspection PyArgumentList
        self._setup_methods[self.method](self)

    def reset(self):
        super().reset()
        # noinspection PyArgumentList
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


class IndexGenerator(Generator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.index = 0

    def frame(self):
        frame = np.arange(self.index, self.index + self.device.framesize)
        self.index += self.device.framesize
        return frame

    def reset(self):
        super().reset()
        self.index = 0
