import numpy as np
from scipy import signal as sps

from . import utils


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
        output_levels, self._buffer = sps.lfilter([self._digital_constant], [1, self._digital_constant - 1], input_levels, zi=self._buffer)
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


# noinspection DuplicatedCode
def deconvolve(exc_sig, rec_sig, fs=None, f_low=None, f_high=None, res_len=None,
               filter_args=None, deconv_type='lin', deconv_phase=True,
               inv_dyn_db=None, win_in_len=None, win_out_len=None):
    """Deconvolve signals.

    Perform signal deconvolution of recording input spectra over excitation
    output spectra.

    Arguments:
        exc_sig (`numpy.ndarray`): ``(..., n_ch, n_samp)`` shape excitation
            output signals.
        rec_sig (`numpy.ndarray`): ``(..., n_ch, n_samp)`` shape recorded
            input signals.
        fs (`float`, optional): sampling frequency in Hertz, default ``None``
        f_low (`float`, optional): lower bandpass (or highpass) cutoff
            frequency in Hertz (or in normalized frequency in case no
            sampling frequency is given), default ``None``.
        f_high (`float`, optional): upper bandpass (or lowpass cutoff
            frequency in Hertz (or in normalized frequency in case no
            sampling frequency is given), default ``None``.
        res_len (`float`, optional): target length of deconvolution results
            in seconds (or in samples in case no sampling frequency is given),
            default ``None`` resembling no truncation.
        filter_args (optional): arguments will be passed to
            `scipy.signal.iirfilter`.
        deconv_type (``'lin'`` or ``'cyc'``, optional): linear deconvolution to
             cut non-harmonic distortion products from the resulting signals
             or cyclic deconvolution otherwise, default ``'lin'``.
        deconv_phase (`bool`, optional): if the phase of the excitation signals
            should be considered (complex deconvolution) or neglected otherwise
            (compensation of the magnitude spectrum), default ``True``.
        inv_dyn_db (`float`, optional): inversion dynamic limitation applied to
            excitation signal in Decibel, default ``None``.
        win_in_len (`int`, optional): length of fade-in window applied to
            deconvolution results in samples, default ``None``.
        win_out_len (`int`, optional): length of fade-out window applied to
            deconvolution results in samples, default ``None``.

    Returns:
        `numpy.ndarray`: ``(..., n_ch, n_samp)`` shape deconvolved signals.
    """
    exc_sig = np.atleast_2d(exc_sig)  # as [..., channels, samples]
    rec_sig = np.atleast_2d(rec_sig)  # as [..., channels, samples]
    if exc_sig.shape[-2] > 1 and exc_sig.shape[-2] != rec_sig.shape[-2]:
        raise ValueError(
                f'Mismatch of provided excitation output ({exc_sig.shape[-2]}) '
                f'vs. recorded input ({rec_sig.shape[-2]}) size.'
        )

    filter_args = filter_args or {}
    filter_args.setdefault('N', 8)
    if fs:
        filter_args.setdefault('fs', fs)
    if f_low not in (None, False) and f_high:
        filter_args.setdefault('Wn', (f_low, f_high))
        filter_args['btype'] = 'bandpass'
    elif f_low not in (None, False):
        filter_args.setdefault('Wn', f_low)
        filter_args['btype'] = 'highpass'
    elif f_high not in (None, False):
        filter_args.setdefault('Wn', f_high)
        filter_args['btype'] = 'lowpass'
    filter_args.setdefault('ftype', 'butter')
    filter_args['output'] = 'sos'

    # # Plot provided signals in time domain
    # import matplotlib.pyplot as plt
    # etc_rec = 20 * np.log10(np.abs(np.squeeze(rec_sig[0])))
    # etc_exc = 20 * np.log10(np.abs(np.squeeze(exc_sig[0])))
    # plt.plot(etc_rec.T, 'k', label='recorded')
    # plt.plot(etc_exc.T, 'r', label='excitation')
    # plt.title('Provided signals')
    # plt.xlabel('Samples')
    # plt.ylabel('Energy Time Curve in dB')
    # plt.legend(loc='best')
    # plt.grid(which='both', axis='both')
    # plt.xlim([0, etc_exc.shape[-1]])
    # plt.ylim([-120, 0]
    #          + np.ceil((np.nanmax(np.hstack((etc_exc, etc_rec))) / 5) + 1) * 5)
    # plt.tight_layout()
    # plt.show()

    # Determine processing length
    n_samples = max(rec_sig.shape[-1], exc_sig.shape[-1])
    if 'lin' in deconv_type.lower():
        n_samples *= 2
    elif 'cyc' not in deconv_type.lower():
        raise ValueError('Unknown deconvolution type `{}`'.format(deconv_type))

    # Transform into frequency domain including desired zero padding
    exc_fd_inv = 1. / np.fft.rfft(exc_sig, n=n_samples, axis=-1)
    rec_fd = np.fft.rfft(rec_sig, n=n_samples, axis=-1)

    # # Plot provided signals in frequency domain
    # import matplotlib.pyplot as plt
    # f = np.fft.rfftfreq(n_samples, d=1. / fs)
    # tf_rec = 20 * np.log10(np.abs(np.squeeze(rec_fd[0])))
    # tf_exc = 20 * np.log10(np.abs(np.squeeze(
    #         np.fft.rfft(exc_sig[0], n=n_samples, axis=-1))))
    # plt.semilogx(f, tf_rec.T, 'k', label='recorded')
    # plt.semilogx(f, tf_exc.T, 'r', label='excitation')
    # plt.title('Provided signals')
    # plt.xlabel('Frequency' + ('in Hz' if fs is not None else ''))
    # plt.ylabel('Magnitude in dB')
    # plt.legend(loc='best')
    # plt.grid(which='both', axis='both')
    # if fs is not None:
    #     plt.xlim([20, fs / 2])
    # plt.ylim([-120, 0]
    #          + np.ceil((np.nanmax(np.hstack((tf_exc, tf_rec))) / 5) + 1) * 5)
    # plt.tight_layout()
    # plt.show()

    # Apply zero phase
    if not deconv_phase:
        exc_fd_inv = np.abs(exc_fd_inv)

    # Apply inversion dynamic limitation
    if inv_dyn_db not in (None, False):
        inv_dyn_min_lin = np.min(np.abs(exc_fd_inv)) * 10 ** (
                    abs(inv_dyn_db) / 20)

        # Determine bins that need to be limited
        in_fd_where = np.abs(exc_fd_inv) > inv_dyn_min_lin

        # Substitute magnitude and leave phase untouched
        exc_fd_inv[..., in_fd_where] = (
                inv_dyn_min_lin
                * np.exp(1j * np.angle(exc_fd_inv[..., in_fd_where]))
        )

        # # Plot effect of dynamic limitation in frequency domain
        # import matplotlib.pyplot as plt
        # f = np.fft.rfftfreq(n_samples, d=1./fs)
        # tf_raw = 20 * np.log10(np.abs(np.squeeze(
        #         1. / np.fft.rfft(exc_sig[0], n=n_samples, axis=-1))))
        # tf_dyn = 20 * np.log10(np.abs(np.squeeze(exc_fd_inv[0])))
        # plt.semilogx(f, tf_raw.T, 'k', label='raw')
        # plt.semilogx(f, tf_dyn.T, 'r', label='limited')
        # plt.title('Effect of dynamic limitation')
        # plt.xlabel('Frequency' + ('in Hz' if fs is not None else ''))
        # plt.ylabel('Magnitude in dB')
        # plt.legend(loc='best')
        # plt.grid(which='both', axis='both')
        # if fs is not None:
        #     plt.xlim(right=fs / 2)
        # plt.ylim([-120, 0]
        #          + np.ceil((np.nanmax(np.hstack((tf_raw, tf_dyn))) / 5) + 1) * 5)
        # plt.tight_layout()
        # plt.show()

    # Deconvolve signals
    res_tf = rec_fd * exc_fd_inv

    # Apply bandpass filter
    if 'Wn' in filter_args:
        filter_sos = sps.iirfilter(**filter_args)
        _, filter_tf = sps.sosfreqz(
                sos=filter_sos,
                worN=res_tf.shape[-1],
                whole=False,
        )
        res_tf *= filter_tf

        # # Plot effect of bandpass filter in frequency domain
        # import matplotlib.pyplot as plt
        # f = np.fft.rfftfreq(n_samples, d=1. / fs)
        # tf_raw = 20 * np.log10(np.abs(np.squeeze((res_tf / filter_tf)[0])))
        # tf_fil = 20 * np.log10(np.abs(np.squeeze(res_tf[0])))
        # plt.semilogx(f, tf_raw.T, 'k', label='raw')
        # plt.semilogx(f, tf_fil.T, 'r', label='filtered')
        # plt.title('Effect of bandpass filter')
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

    # Determine result length
    if 'lin' in deconv_type.lower():
        n_samples /= 2
    if res_len is not None and res_len != 0:
        n_samples = res_len * (fs if fs is not None else 1)
    n_samples = int(n_samples)

    # Transform into time domain and truncate to target length
    res_ir = np.fft.irfft(res_tf, axis=-1)[..., :n_samples]
    # res_ir_copy = res_ir.copy()  # solely for plotting purposes

    # # Plot effect of result signals truncation in time domain
    # if res_len is not None and res_len != 0:
    #     import matplotlib.pyplot as plt
    #     etc_ful = 20 * np.log10(np.abs(np.squeeze(np.fft.irfft(res_tf[0], axis=-1))))
    #     etc_tar = 20 * np.log10(np.abs(np.squeeze(res_ir_copy[0])))
    #     plt.plot(etc_ful.T, 'k', label='full')
    #     plt.plot(etc_tar.T, 'r', label='target')
    #     plt.title('Effect of result signals truncation')
    #     plt.xlabel('Samples')
    #     plt.ylabel('Energy Time Curve in dB')
    #     plt.legend(loc='best')
    #     plt.grid(which='both', axis='both')
    #     plt.xlim([0, etc_ful.shape[-1]])
    #     plt.ylim([-120, 0]
    #              + np.ceil((np.nanmax(np.hstack((etc_ful, etc_tar))) / 5) + 1) * 5)
    #     plt.tight_layout()
    #     plt.show()

    # Apply start and end window
    res_ir = utils.fade_signal(
            sig=res_ir,
            win_in_len=win_in_len,
            win_out_len=win_out_len,
    )

    # # Plot effect of result signals windowing in time domain
    # import matplotlib.pyplot as plt
    # etc_raw = 20 * np.log10(np.abs(np.squeeze(res_ir_copy[0])))
    # etc_win = 20 * np.log10(np.abs(np.squeeze(res_ir[0])))
    # plt.plot(etc_raw.T, 'k', label='raw')
    # if ((win_in_len is not None and win_in_len != 0)
    #         or (win_out_len is not None and win_out_len != 0)):
    #     plt.plot(etc_win.T, 'r', label='windowed')
    # plt.title('Result signals')
    # plt.xlabel('Samples')
    # plt.ylabel('Energy Time Curve in dB')
    # plt.legend(loc='best')
    # plt.grid(which='both', axis='both')
    # plt.xlim([0, res_ir.shape[-1]])
    # plt.ylim([-120, 0]
    #          + np.ceil((np.nanmax(np.hstack((etc_raw, etc_win))) / 5) + 1) * 5)
    # plt.tight_layout()
    # plt.show()
    #
    # # Plot result signals in frequency domain
    # import matplotlib.pyplot as plt
    # f = np.fft.rfftfreq(n_samples, d=1. / fs)
    # tf_res = 20 * np.log10(np.abs(np.squeeze(np.fft.rfft(res_ir, axis=-1))))
    # plt.semilogx(f, tf_res.T)
    # plt.title('Result signals')
    # plt.xlabel('Frequency' + ('in Hz' if fs is not None else ''))
    # plt.ylabel('Magnitude in dB')
    # plt.grid(which='both', axis='both')
    # if fs is not None:
    #     plt.xlim([20, fs / 2])
    # plt.ylim([-120, 0] + np.ceil((np.nanmax(tf_res) / 5) + 1) * 5)
    # plt.tight_layout()
    # plt.show()

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
