import numpy as np


def _apply_to_signals(func, signals):
    if isinstance(signals, np.ndarray):
        return func(signals)
    return tuple(func(signal) for signal in signals)


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

    if samplerate is not None:
        if pre_pad is not None:
            pre_pad = round(samplerate * pre_pad)
        if post_pad is not None:
            post_pad = round(samplerate * post_pad)
    pre_pad = pre_pad or 0
    post_pad = post_pad or 0
    pre_pad = np.zeros(signals.shape[:-1] + (pre_pad,))
    post_pad = np.zeros(signals.shape[:-1] + (post_pad,))

    def pad(signal):
        return np.concatenate([pre_pad, signal, post_pad], axis=-1)

    return _apply_to_signals(pad, signals)


def fade_signals(signals, fade_in=None, fade_out=None, samplerate=None, inplace=True):
    if fade_in is None and fade_out is None:
        return signals

    if samplerate is not None:
        if fade_in is not None:
            fade_in = round(samplerate * fade_in)
        if fade_out is not None:
            fade_out = round(samplerate * fade_out)

    if fade_in is None:
        fade_in = np.ones(shape=0)
    else:
        fade_in = np.sin(np.linspace(0, np.pi / 2, fade_in))**2
    if fade_out is None:
        fade_out = np.ones(shape=0)
    else:
        fade_out = np.sin(np.linspace(np.pi / 2, 0, fade_out))**2

    def fade(signal):
        # Apply the fade in and the fade out
        # Make sure to respect the `inplace` variable!
        if not inplace:
            signal = signal.copy()
        signal[..., :fade_in.size] *= fade_in
        signal[..., -fade_out.size:] *= fade_out
        return signal

    return _apply_to_signals(fade, signals)
