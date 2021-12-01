import numpy as np
from numpy.fft import rfft as fft, irfft as ifft
import scipy.signal
import queue
import warnings
import functools
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
            self.device._Device__generators.remove(self)
        self._device = dev
        if self.device is not None:
            # Register to the new device
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
    """Repeated generation of arbritrary signals.

    Implementation of `Generator` for arbritrary signals.

    Arguments:
        repetitions (`float`): The number of cycles to output before stopping, default `np.inf`.
        **kwargs: Will be saved as ``kwargs`` and accessible in `setup`.
    Keyword Arguments:
        signal (`numpy.ndarray`): One cycle of the signal to output.
    """
    def __init__(self, repetitions=np.inf, fade_in=0, fade_out=0, pre_pad=0, post_pad=0, **kwargs):
        super().__init__(**kwargs)
        self.repetitions = repetitions  # Default to continious output
        self.fade_in = fade_in
        self.fade_out = fade_out
        self.pre_pad = pre_pad
        self.post_pad = post_pad
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

    @property
    def signal(self):
        return self._signal
    
    @signal.setter
    def signal(self, sig):
        sig = np.array(sig) # Makes sure the input is a numpy array and makes a copy
 
        # Checks the max length of the applied fades.
        # If someone requests a longer fade than the signal, let them.
        # The only thing that happens in that a fade in will not reach 1, and a fade out will start below 1
        fade_in_samples = int(self.fade_in * self.device.fs)
        if fade_in_samples > 0:
            fade_in = np.sin(np.linspace(0, np.pi/2, fade_in_samples))**2
            fade_in_samples = min(fade_in_samples, sig.shape[-1])
            sig[..., :fade_in_samples] *= fade_in[:fade_in_samples]

        fade_out_samples = int(self.fade_out * self.device.fs)
        if fade_out_samples > 0:
            fade_out = np.sin(np.linspace(np.pi/2, 0, fade_out_samples))**2
            fade_out_samples = min(fade_out_samples, sig.shape[-1])
            sig[..., -fade_out_samples:] *= fade_out[-fade_out_samples:]

        pre_pad_samples = int(self.pre_pad * self.device.fs)
        post_pad_samples = int(self.post_pad * self.device.fs)
        pre_pad = np.zeros(sig.shape[:-1] + (pre_pad_samples,))
        post_pad = np.zeros(sig.shape[:-1] + (post_pad_samples,))

        self._signal = np.concatenate([pre_pad, sig, post_pad])


    def setup(self):
        """Configures the signal.

        Create the signal manually and pass it while creating the generator
        as the ``signal`` argument.

        It is possible to inherit `ArbitrarySignalGenerator` and override the
        setup method. Create one cycle of the signal and store it in
        ``self.signal``. Access the underlying device as ``self.device``,
        which has important properties, e.g. samplerate ``fs``.
        All keyword arguments passed while creating instances are available
        as ``self.key``.

        Note:
            Call `ArbitrarySignalGenerator.setup(self)` from subclasses.
        """
        super().setup()

    @property
    def duration(self):
    	return len(self.signal) / self.device.fs * self.repetitions


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
    def __init__(self, start_frequency, stop_frequency, sweep_length,
                 method='logarithmic', bidirectional=False, **kwargs):
        super().__init__(**kwargs)
        self.start_frequency = start_frequency
        self.stop_frequency = stop_frequency
        self.sweep_length = sweep_length
        self.method = method
        self.bidirectional = bidirectional

        def deconvolve(self, *args, **kwargs):
            if len(args) == 2:
                kwargs['reference'] = args[0]
                kwargs['output'] = args[1]
            if len(args) == 1:
                kwargs['output'] = args[0]
            if 'reference' not in kwargs:
                kwargs['reference'] = self.signal
                kwargs.setdefault('method', 'analytic')
            kwargs.setdefault('fs', self.device.fs)
            kwargs.setdefault('f_low', min(self.start_frequency, self.stop_frequency))
            kwargs.setdefault('f_high', max(self.start_frequency, self.stop_frequency))
            return SweepGenerator.deconvolve(**kwargs)
        self.deconvolve = functools.partial(deconvolve, self)

    def setup(self):
        super().setup()
        time_vector = np.arange(round(self.sweep_length * self.device.fs)) / self.device.fs
        phase_rate = self.sweep_length / np.log(self.stop_frequency / self.start_frequency)
        phase = 2 * np.pi * self.start_frequency * phase_rate * np.exp(time_vector / phase_rate)
        signal = np.sin(phase)
        if self.bidirectional:
            signal = np.concatenate([signal, signal[::-1]])
        last_zero_crossing = -1
        while np.sign(signal[last_zero_crossing-1]) == np.sign(signal[-1]):
            last_zero_crossing -= 1
        signal[last_zero_crossing:] = 0
        self.signal = signal

    @classmethod
    def deconvolve(cls, reference, output, fs=None, f_low=None, f_high=None, T=None, fade_out=None, method='regularized', orders=1, **kwargs):
        if method == 'filtering':
            inverse_filter = cls._filtered_inverse(reference, fs=fs, f_low=f_low, f_high=f_high, **kwargs)
        elif method == 'regularized':
            inverse_filter = cls._regularized_inverse(reference, fs=fs, f_low=f_low, f_high=f_high, **kwargs)
        elif method == 'analytic':
            inverse_filter = cls._analytic_inverse(reference, fs=fs, f_low=f_low, f_high=f_high, **kwargs)
        elif method=='time-reversal':
            inverse_filter = cls._time_reversal_filter(reference, fs=fs, f_low=f_low, f_high=f_high, **kwargs)
        else:
            raise ValueError('Unknown method `{}`'.format(method))

        output = np.asarray(output)
        meas_length = output.shape[-1]
        TF = np.fft.rfft(output, n=2*meas_length, axis=-1) * np.fft.rfft(inverse_filter, n=2*meas_length)

        ir_whole = np.fft.irfft(TF, axis=-1)
        if orders == 'whole':
            return np.squeeze(ir_whole)
    

        ir = [ir_whole[..., inverse_filter.size:]]
        if orders > 1:
            phase_rate = inverse_filter.size / np.log(f_high / f_low)
            for order in range(1, orders):
                ir.append(ir_whole[..., inverse_filter.size - int(phase_rate * np.log(order + 1)):inverse_filter.size - int(phase_rate * np.log(order))])

        if T is not None:
            if fs is not None:
                T = T * fs
            for order in range(orders):
                ir[order] = ir[order][..., :int(T)]
        
        if fade_out is not None and fade_out>0:
            if fs is not None:
                fade_out_samples = min(int(fade_out * fs), ir.shape[-1])
            fade_out = np.sin(np.linspace(np.pi/2, 0, fade_out_samples))**2
            for order in range(orders):
                ir[order][..., -fade_out_samples:] *= fade_out

        for order in range(orders):
            ir[order] = np.squeeze(ir[order])

        if len(ir) == 1:
            ir = ir[0]
        return ir

    @classmethod
    def _filtered_inverse(cls, reference, fs=None, f_low=None, f_high=None, **kwargs):
        inverse_spectrum = 1 / np.fft.rfft(reference)

        filter_args = kwargs or {}
        filter_args.setdefault('N', 8)
        if f_low is not None and f_high is not None:
            if fs is not None:
                f_low = f_low / fs * 2
                f_high = f_high / fs * 2
            filter_args.setdefault('Wn', (f_low, f_high))
            filter_args['btype'] = 'bandpass'
        elif f_low is not None:
            if fs is not None:
                f_low = f_low / fs * 2
            filter_args.setdefault('Wn', f_low)
            filter_args['btype'] = 'highpass'
        elif f_high is not None:
            if fs is not None:
                f_high = f_high / fs * 2
            filter_args.setdefault('Wn', f_high)
            filter_args['btype'] = 'lowpass'
        
        filter_args.setdefault('ftype', 'butter')
        filter_args['output'] = 'sos'
        if 'Wn' in filter_args:
            sos = scipy.signal.iirfilter(**filter_args)
            _, H = scipy.signal.sosfreqz(sos, inverse_spectrum.size)
            inverse_spectrum = inverse_spectrum * H
        inverse_filter = np.fft.irfft(inverse_spectrum)
        return inverse_filter

    @classmethod
    def _regularized_inverse(cls, reference, fs=None, f_low=None, f_high=None, width=1/12,
                             interior_regularization=0.01, exterior_regulrization=1, **kwargs):
        ref_spectrum = np.fft.rfft(reference)
        fs = fs or 1
        f_low = f_low or 0
        f_high = f_high or fs/2
        f_low_interior = f_low * 2**width
        f_high_interior = f_high / 2**width
        f = np.fft.rfftfreq(reference.size, 1/fs)

        regularization = np.zeros(ref_spectrum.shape)
        interior_idx = (f_low_interior <= f) & (f <= f_high_interior)
        regularization[interior_idx] = 1/f[interior_idx] * interior_regularization
        regularization[0] = 0
        
        regularization[f<=f_low] = exterior_regulrization / f_low
        slope_idx = (f_low<f) & (f<f_low_interior)
        regularization[slope_idx] = np.geomspace(exterior_regulrization / f_low, interior_regularization / f_low_interior, np.sum(slope_idx))

        slope_idx = (f_high_interior<f) & (f<f_high)
        regularization[f_high<=f] = exterior_regulrization / f_high
        regularization[slope_idx] = np.geomspace(interior_regularization / f_high_interior, exterior_regulrization / f_high, np.sum(slope_idx))
        
        regularization *= np.mean(np.abs(ref_spectrum[interior_idx])**2 / regularization[interior_idx]) * interior_regularization
        inverse_spectrum = ref_spectrum.conj() / (ref_spectrum.conj() * ref_spectrum + regularization)
        inverse_filter = np.fft.irfft(inverse_spectrum)
        return inverse_filter

    @classmethod
    def _analytic_inverse(cls, reference, fs, f_low=None, f_high=None, **kwargs):
        T = reference.size / fs
        phase_rate = T / np.log(f_high / f_low)
        f = np.fft.rfftfreq(reference.size, 1/fs)
        with np.errstate(divide='ignore', invalid='ignore'):
            inverse_spectrum = 2 * (f / phase_rate)**0.5 * np.exp(-2j * np.pi * f * phase_rate * (1 - np.log(f / f_low)) + 1j * np.pi / 4)
        inverse_spectrum[0] = 0
        inverse_filter = np.fft.irfft(inverse_spectrum) / fs
        return inverse_filter

    @classmethod
    def _time_reversal_filter(cls, reference, fs, f_low, f_high, **kwargs):
        amplitude_correction = 10**(np.linspace(0, -6 * np.log2(f_high/f_low), reference.size) / 20)
        inverse_filter = reference[-1::-1] * amplitude_correction
        return inverse_filter


class MaximumLengthSequenceGenerator(ArbitrarySignalGenerator):
    """Generation of maximum length sequences.

    Arguments:
        order (`int`): The order or the sequence. The total length  is ``2**order - 1``.
        repetitions (`float`, optional): The number of repetitions, default `np.inf`.
    See Also:
        `ArbitrarySignalGenerator`, `scipy.signal.max_len_seq`
    """
    def __init__(self, order, **kwargs):
        super().__init__(**kwargs)
        self.order = order

    def setup(self):
        super().setup()
        self.sequence, state = scipy.signal.max_len_seq(self.order)
        self.signal = (1 - 2 * self.sequence).astype('float64')

    def analyze(self, data):
	    data = np.atleast_2d(data)
	    seq_len = 2**self.order - 1
	    # order = np.round(np.log2(seq_len+1)).astype(int)
	    reps = int(data.shape[1] / seq_len)
	    
	    if reps > 1:
	        response = data[:, seq_len:reps * seq_len].reshape((-1, reps-1, seq_len)).mean(axis=1)
	    else:
	        response = data[:, :seq_len]
	    
	    ps = np.zeros(seq_len, dtype=np.int64)
	    for idx in range(seq_len):
	        for s in range(self.order):
	            ps[idx] += self.sequence[(idx-s)%seq_len] << (self.order-1-s)
	    
	    indices = np.argsort(ps)[2**np.arange(self.order-1, -1, -1)-1]
	    pl = np.zeros(seq_len, dtype=np.int64)
	    for idx in range(seq_len):
	        for s in range(self.order):
	            pl[idx] += self.sequence[(indices[s]-idx)%seq_len] << (self.order-s-1)
	            
	    transform = np.zeros((data.shape[0], seq_len+1))
	    transform[:, ps] = response
	    for _ in range(self.order):
	        transform = np.concatenate((transform[:, ::2] + transform[:, 1::2], transform[:,::2] - transform[:, 1::2]), axis=1)
	    ir = transform[:, pl] / (seq_len+1)
	    return np.squeeze(ir)


class FunctionGenerator(Generator):
    """Generates signals from a shape function.

    Implementation of `Generator` for standard funcitons.

    Arguments:
        frequency (`float`): The frequecy of the signal, in Hz.
        repetitions (`float`, optional): The number of repetitions, default `np.inf`.
        shape (`str`, optional): Function shape, default ``'sine'``. Currently
            available functions are

                - ``'sine'``: `numpy.sin`
                - ``'sawtooth'``: `scipy.signal.sawtooth`
                - ``'square'``: `scipy.signal.square`

        phase_offset (`float`, optional): Phase offset of the signal in radians, default 0.
        shape_kwargs (`dict`): Keyword arguments for shape function.
    """
    _functions = {
        'sin': scipy.signal.waveforms.sin,
        'saw': scipy.signal.waveforms.sawtooth,
        'squ': scipy.signal.waveforms.square
    }

    def __init__(self, frequency, repetitions=np.inf,
                 shape='sine', phase_offset=0, shape_kwargs=None, **kwargs):
        super().__init__(**kwargs)
        self.repetitions = repetitions  # Default to continious output
        self.frequency = frequency
        self.shape = shape
        self.phase_offset = phase_offset
        self.shape_kwargs = {} if shape_kwargs is None else shape_kwargs

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
        super().setup()
        self._setup_methods[self.method](self)

    def reset(self):
        super().reset()
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
