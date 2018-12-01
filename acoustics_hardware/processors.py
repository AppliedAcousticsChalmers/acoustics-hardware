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
        """Processes a single fraame of input.

        The input frame might be the rame object as the read frame, so a
        processor should not manitulate the data in place.

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
        return self._buffer**0.5


def deconvolve(input, output, fs=None, f_low=None, f_high=None, T=None, filter_args=None):
    output = np.atleast_2d(output)
    input = np.pad(input, (0, output.shape[1] - input.shape[0]), mode='constant')
    TF = np.fft.rfft(output, axis=1) / np.fft.rfft(input)

    filter_args = filter_args or {}
    filter_args.setdefault('N', 8)
    if f_low is not None and f_high is not None:
        if fs is not None:
            f_low = f_low / fs * 2
            f_high = f_high / fs * 2
        filter_args.setdefault('Wn', (f_low, f_high))
    filter_args.setdefault('ftype', 'butter')
    filter_args['btype'] = 'bandpass'
    filter_args['output'] = 'sos'
    if 'Wn' in filter_args:
        sos = scipy.signal.iirfilter(**filter_args)
        _, H = scipy.signal.sosfreqz(sos, TF.shape[1])
        TF = TF * H
    ir = np.fft.irfft(TF, axis=1)
    if T is not None:
        if fs is not None:
            T = T * fs
        ir = ir[:, :int(T)]
    return np.squeeze(ir)


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
            ps[idx] += reference[(idx-s)%seq_len] << (order-1-s)
    
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