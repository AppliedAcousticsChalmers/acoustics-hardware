import numpy as np
import fractions


def _apply_to_signals(func, signals):
    if isinstance(signals, np.ndarray):
        return func(signals)
    return tuple(func(signal) for signal in signals)


def _length_parser(length, samplerate):
    if isinstance(length, str):
        if '%' in length:
            length = float(length.strip('%')) / 100
            def parser(signal):
                return round(signal.shape[-1] * length)
        else:
            length = fractions.Fraction(length)
            def parser(signal):
                return round(signal.shape[-1] * length)
    else:
        length = length or 0
        if samplerate is not None:
            length = round(samplerate * length)
        def parser(signal):
            return length
    return parser


def truncate_signals(signals, length=None, samplerate=None):
    """Cut signals down to a specified length."""
    if length is None:
        return signals
    if samplerate is not None:
        length = round(samplerate * length)

    def truncation(signal):
        return signal[..., :length]

    return _apply_to_signals(truncation, signals)


def extend_signals(signals, length=None, samplerate=None):
    """Zero-pad signals to a certain length.

    Note that this will not shorten signals, so if they are longer
    than the requested length they will be unchanged.
    """
    if length is None:
        return signals
    if samplerate is not None:
        length = round(samplerate * length)

    def extend(signal):
        padding = length - signal.shape[-1]
        if padding < 1:
            return signal.copy()
        padding = np.zeros(signal.shape[:-1] + (padding,))
        padded = np.concatenate([signal, padding], axis=-1)
        return padded

    return _apply_to_signals(extend, signals)


def pad_signals(signals, pre_pad=None, post_pad=None, samplerate=None):
    if pre_pad is None and post_pad is None:
        return signals

    pre_pad = _length_parser(pre_pad, samplerate)
    post_pad = _length_parser(post_pad, samplerate)
    def pad(signal):
        pre = np.zeros(signal.shape[:-1] + (pre_pad(signal),))
        post = np.zeros(signal.shape[:-1] + (post_pad(signal),))
        return np.concatenate([pre, signal, post], axis=-1)

    return _apply_to_signals(pad, signals)


def fade_signals(signals, fade_in=None, fade_out=None, samplerate=None, inplace=False):
    if fade_in is None and fade_out is None:
        return signals

    fade_in = _length_parser(fade_in, samplerate)
    fade_out = _length_parser(fade_out, samplerate)

    def fade(signal):
        if not inplace:
            signal = signal.copy()
        in_samples = fade_in(signal)
        out_samples = fade_out(signal)
        signal[..., :in_samples] *= np.sin(np.linspace(0, np.pi / 2, in_samples))**2
        signal[..., -out_samples:] *= np.sin(np.linspace(np.pi / 2, 0, out_samples))**2
        return signal

    return _apply_to_signals(fade, signals)


def nonzero_signals(signals, inplace=False):
    def nonzero(signal):
        if not inplace:
            signal = signal.copy()
        zero_indices = signal == 0
        signal[zero_indices] = np.min(np.abs(signal[np.invert(zero_indices)]))
        return signal
    return _apply_to_signals(nonzero, signals)
