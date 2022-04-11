import numpy as np


def truncate_signals(*signals, length=None, samplerate=None):
    if length is None:
        return signals
    if samplerate is not None:
        length = round(samplerate * length)
    return [signal[..., :length] for signal in signals]


def pad_signals(*signals, length=None, samplerate=None):
    if length is None:
        return signals
    if samplerate is not None:
        length = round(samplerate * length)
    padded_signals = []
    for signal in signals:
        padding = length - signal.shape[-1]
        padding = np.zeros(signal.shape[:-1] + (padding,))
        padded = np.concatenate([signal, padding], axis=-1)
        padded_signals.append(padded)
    return padded_signals


def fade_signals(*signals, fade_in=None, fade_out=None, samplerate=None, inplace=True):
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

    def apply_fades(signal):
        # Apply the fade in and the fade out
        # Make sure to respect the `inplace` variable!
        if not inplace:
            signal = signal.copy()
        signal[..., :fade_in.size] *= fade_in
        signal[..., -fade_out.size:] *= fade_out
        return signal

    return [apply_fades(signal) for signal in signals]
