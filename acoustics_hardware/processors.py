import numpy as np
import scipy.signal


class Processor:
    """Base class for processors

    A processor is an object that manipulates the data in some way.
    """
    def __init__(self, device=None, **kwargs):
        self.device = device
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __call__(self, frame):
        return self.process(frame)

    def process(self, frame):
        """Processes a single frame of input.

        The input frame might be the same object as the read frame, so a
        processor should not manipulate the data in place.

        Arguments:
            frame (`numpy.ndarray`): ``(n_ch, n_samp)`` shape input frame.
        """

    def setup(self):
        pass

    def reset(self):
        pass

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device):
        self._device = device


class LevelDetector(Processor):
    """Single channel level detector.

    A level detector that tracks the root-mean-square level in a signal.
    The level is tracked with an exponentially decaying time average,
    implemented as a low-passed squared amplitude.

    Arguments:
        channel (`int`): The index of the channel to track.
        time_constant (`float`): Time constant for the exponential decay in
            seconds, default 50 ms.
        **kwargs: Extra arguments will be passed to `Processor`.
    """
    def __init__(self, channel, time_constant=50e-3, **kwargs):
        super().__init__(**kwargs)
        self.time_constant = time_constant

        self.channel = channel
        self.reset()

    def process(self, frame):
        # Squared input level is tracked in order for the RMS trigger to work properly.
        input_levels = frame[self.channel]**2
        output_levels, self._buffer = scipy.signal.lfilter([self._digital_constant], [1, self._digital_constant - 1], input_levels, zi=self._buffer)
        return output_levels**0.5

    def reset(self):
        super().reset()
        self._buffer = np.atleast_1d(0)

    @property
    def time_constant(self):
        return -1 / (np.log(1 - self._digital_constant) * self.device.fs)

    @time_constant.setter
    def time_constant(self, val):
        self._digital_constant = 1 - np.exp(-1 / (val * self.device.fs))

    @property
    def current_level(self):
        return self._buffer[0]**0.5


def deconvolve(exc_sig, rec_sig, fs=None, f_low=None, f_high=None, res_len=None,
               filter_args=None):
    """Deconvolve signals.

    Perform signal deconvolution of output spectrum over input spectrum.

    Arguments:
        exc_sig (`numpy.ndarray`): ``(n_ch, n_samp)`` shape excitation output
            signal.
        rec_sig (`numpy.ndarray`): ``(n_ch, n_samp)`` shape recorded input
            signal.
        fs (`float`): sampling frequency in Hertz, default ``None``
        f_low (`float`): lower bandpass cutoff frequency in Hertz
            (or in normalized frequency in case no sampling frequency is given),
            default ``None`` resembling no filter.
        f_high (`float`): upper bandpass cutoff frequency in Hertz
            (or in normalized frequency in case no sampling frequency is given),
            default ``None`` resembling no filter.
        res_len (`float`): target length of deconvolution results in seconds
            (or in samples in case no sampling frequency is given),
            default ``None`` resembling no truncation.
        filter_args: arguments will be passed to `scipy.signal.iirfilter`.

    Returns:
        `numpy.ndarray`: ``(n_ch, n_samp)`` shape deconvolved signal.
    """
    rec_sig = np.atleast_2d(rec_sig)
    filter_args = filter_args or {}
    filter_args.setdefault('N', 8)
    if fs is not None:
        filter_args.setdefault('fs', fs)
    if f_low is not None and f_high is not None:
        filter_args.setdefault('Wn', (f_low, f_high))
    filter_args.setdefault('ftype', 'butter')
    filter_args['btype'] = 'bandpass'
    filter_args['output'] = 'sos'

    # Zero pad excitation signal to size of recorded signal
    exc_sig = np.pad(exc_sig, pad_width=(0, rec_sig.shape[-1] - exc_sig.shape[-1]))

    # Deconvolve signals
    res_tf = np.fft.rfft(rec_sig, axis=-1) / np.fft.rfft(exc_sig, axis=-1)

    # Apply bandpass filter
    if 'Wn' in filter_args:
        sos = scipy.signal.iirfilter(**filter_args)
        _, filter_tf = scipy.signal.sosfreqz(sos, worN=res_tf.shape[-1], whole=False)
        res_tf *= filter_tf

        # # Plot effect of bandpass filter in frequency domain
        # import matplotlib.pyplot as plt
        # f = np.fft.rfftfreq(rec_sig.shape[-1], d=1. / fs)
        # tf_raw = 20 * np.log10(np.abs(np.squeeze(res_tf / filter_tf)[0]))
        # tf_fil = 20 * np.log10(np.abs(np.squeeze(res_tf[0])))
        # plt.semilogx(f, tf_raw.T, 'k', label='raw')
        # plt.semilogx(f, tf_fil.T, 'r', label='filtered')
        # plt.title('Provided signals')
        # plt.xlabel('Frequency' + ('in Hz' if fs is not None else ''))
        # plt.ylabel('Magnitude in dB')
        # plt.legend(loc='best')
        # plt.grid(which='both', axis='both')
        # if fs is not None:
        #     plt.xlim([20, fs / 2])
        # plt.ylim([-120, 0]
        #          + np.ceil((np.nanmax(np.hstack((tf_raw, tf_fil))) / 5) + 1) * 5)
        # plt.tight_layout()
        # plt.show()

    # Transform into time domain
    res_ir = np.fft.irfft(res_tf, axis=-1)

    # Truncate impulse response
    if res_len is not None and res_len != 0:
        if fs is not None:
            res_len *= fs
        res_ir = res_ir[..., :int(res_len)]

    return res_ir


def mls_analysis(reference, output):
    output = np.atleast_2d(output)
    seq_len = len(reference)
    order = np.round(np.log2(seq_len+1)).astype(int)
    reps = int(output.shape[1] / seq_len)
    
    if reps > 1:
        response = output[:, seq_len:reps * seq_len].reshape((-1, reps-1, seq_len)).mean(axis=1)
    else:
        response = output[:, :seq_len]
    
    ps = np.zeros(seq_len, dtype=np.int64)
    for idx in range(seq_len):
        for s in range(order):
            ps[idx] += reference[(idx-s) % seq_len] << (order-1-s)
    
    indices = np.argsort(ps)[2**np.arange(order-1, -1, -1)-1]
    pl = np.zeros(seq_len, dtype=np.int64)
    for idx in range(seq_len):
        for s in range(order):
            pl[idx] += reference[(indices[s]-idx)%seq_len] << (order-s-1)
            
    transform = np.zeros((output.shape[0], seq_len+1))
    transform[:, ps] = response
    for _ in range(order):
        transform = np.concatenate((transform[:, ::2] + transform[:, 1::2], transform[:,::2] - transform[:, 1::2]), axis=1)
    ir = transform[:, pl] / (seq_len+1)
    return np.squeeze(ir)
