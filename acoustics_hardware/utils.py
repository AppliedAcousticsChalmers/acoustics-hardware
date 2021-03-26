import queue

import numpy as np
from scipy.signal.windows import cosine


def flush_Q(q):
    while True:
        try:
            q.get(timeout=0.1)
        except queue.Empty:
            break


def concatenate_Q(q):
    data_list = []
    while True:
        try:
            data_list.append(q.get(timeout=0.1))
        except queue.Empty:
            break
    return np.concatenate(data_list, axis=-1)


def fade_signal(sig, win_in_len=0, win_out_len=0):
    """
    Apply a fade-in and / or fade-out to the signal in the form of one-sided
    cosine squared windows.

    Args:
        sig (`numpy.ndarray`): ``(..., samples)`` shape time domain signal(s).
        win_in_len (`int`, optional): length of fade-in window applied to the
            signal in samples, default 0.
        win_out_len (`int`, optional): length of fade-out window applied to the
            signal in samples, default 0..

    Returns:
        `numpy.ndarray`: ``(..., samples)`` shape time domain signal(s).
    """
    if win_in_len:
        win_in_len = int(win_in_len)  # in samples
        win = cosine(M=win_in_len * 2, sym=True) ** 2
        sig[..., :win_in_len] *= win[:win_in_len]

    if win_out_len:
        win_out_len = int(win_out_len)  # in samples
        win = cosine(M=win_out_len * 2, sym=True) ** 2
        sig[..., -win_out_len:] *= win[win_out_len:]

    return sig
