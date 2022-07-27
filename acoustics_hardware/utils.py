import queue
import sys
from datetime import datetime

import numpy as np

# noinspection PyPackageRequirements
from matplotlib import pyplot as plt
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


def generate_time_estimation_string(duration, is_prediction=True):
    end_time = datetime.now()
    if is_prediction:
        end_time += duration
    return f'{duration} (completed at around {end_time.strftime("%H:%M")})'


@np.errstate(divide="ignore")
def generate_levels_string(data):
    _PEAK_THRESHOLD = -3.0  # in dB

    data_peak = 20 * np.log10(np.max(np.abs(data)))
    data_rms = np.sqrt(np.mean(np.power(data, 2), axis=-1))  # linear
    rms_str = np.array2string(
        20 * np.log10(np.abs(data_rms)),  # in dB
        precision=1,
        sign="+",
        floatmode="fixed",
        separator=", ",
    )

    if data_peak > _PEAK_THRESHOLD:
        peak_str = np.array2string(
            20 * np.log10(np.max(np.abs(data), axis=-1)),  # in dB
            precision=1,
            sign="+",
            floatmode="fixed",
            separator=", ",
        )
        print(
            f"Detected peak above {_PEAK_THRESHOLD:+.1f} dB_FS in {peak_str} dB",
            file=sys.stderr,
        )

    return f"PEAK {data_peak:+.1f} dB, RMS {rms_str} dB"


def plot_grid(coords_az_el_r_rad, title=None, is_show_lines=False):
    # noinspection PyShadowingNames
    def sph2cart(az, el, r):
        r_cos_theta = r * np.cos(el)
        x = r_cos_theta * np.cos(az)
        y = r_cos_theta * np.sin(az)
        z = r * np.sin(el)
        return x, y, z

    # as [azimuth, elevation, radius] in rad
    coords_az_el_r_rad = np.atleast_2d(coords_az_el_r_rad)
    x, y, z = sph2cart(
        coords_az_el_r_rad[0], coords_az_el_r_rad[1], coords_az_el_r_rad[2]
    )  # as (x, y, z) in ms

    plt.figure(title, figsize=(8, 8))
    ax = plt.gca(projection="3d")

    if is_show_lines:
        ax.plot3D(x, y, z)
        for pos in range(len(x)):
            ax.text(
                x[pos],
                y[pos],
                z[pos],
                s=pos,
                fontsize=14,
                horizontalalignment="center",
                verticalalignment="center",
            )
    else:
        ax.scatter3D(x, y, z, s=30, depthshade=True)

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(azim=-160, elev=25)  # in deg

    plt.tight_layout()
    plt.show()


@np.errstate(divide="ignore")
def plot_data(data, fs, ch_labels=None, is_stacked=False, is_etc=True):
    # in dB, second value if `is_stacked` or first value otherwise
    _MAG_DR = [80, 120]
    # in dB, second value if `is_stacked` or first value otherwise
    _ETC_DR = [50, 100]

    data = np.atleast_2d(data)
    if not np.count_nonzero(data) or np.isnan(data).all():
        sys.exit("Signal is all zeros or NaNs (plotting aborted).")
    if ch_labels is not None:
        ch_labels = np.atleast_1d(ch_labels)
        if len(ch_labels) != data.shape[-2]:
            sys.exit(
                f"Mismatch in number of audio channels ({data.shape[-2]}) "
                f"and number of provided labels {len(ch_labels)}."
            )
    if data.ndim == 3:
        # Combine first and second dimension
        # also works for string lists
        ch_labels = np.repeat(ch_labels, data.shape[0])
        data = np.reshape(data, (data.shape[0] * data.shape[1], -1), order="F")
        # as [[[channel0_iter0, samples], [channel0_iter1, samples], ...]

    n_ch, n_sample = data.shape
    t_data = np.linspace(0, n_sample / fs, n_sample)  # in samples
    time_data = 20 * np.log10(np.abs(data)) if is_etc else data
    f_data = np.fft.rfftfreq(n_sample, 1 / fs)  # in Hz
    spec_data = 20 * \
        np.log10(np.abs(np.fft.rfft(data)) / np.sqrt(fs * n_sample))
    # in dB, scaled according to the DFT power spectral density for broadband
    # signals

    if is_stacked:
        n_ch = 1

    fig, axes = plt.subplots(
        nrows=n_ch,
        ncols=2 if is_stacked else 3,
        squeeze=False,
        sharex="col",
        figsize=(15, 5 if is_stacked else (3 * n_ch)),
    )
    for ch in np.arange(n_ch):  # individual channels
        ch_plot = range(data.shape[0]) if is_stacked else ch

        # Time domain
        ax = axes[ch, 0]
        ax.plot(t_data, time_data[ch_plot, :].T)
        ax.set_xlim(0, n_sample / fs)  # in s

        ax.grid()
        if ch_labels is not None:
            label = (
                [f"in_{i}" for i in ch_labels]
                if is_stacked
                else [f"in_{ch_labels[ch]}"]
            )
            ax.legend(label, loc="upper right")
        ax.set_xlabel("Time (s)" if ch >= n_ch - 1 else "")
        if is_etc:
            y_max = np.ceil((time_data[ch_plot, :].max() / 5) + 1) * 5
            ax.set_ylim(
                y_max - (_ETC_DR[1] if is_stacked else _ETC_DR[0]), y_max
            )  # in dB
            ax.set_ylabel("Magnitude (dB)")
        else:
            ax.set_ylabel("Amplitude")

        # Frequency domain
        y_max = np.ceil((spec_data[ch_plot, :].max() / 5) + 1) * 5
        ax = axes[ch, 1]
        ax.semilogx(f_data, spec_data[ch_plot, :].T)
        ax.set_xlim(20, fs / 2)  # in Hz
        ax.set_ylim(y_max -
                    (_MAG_DR[1] if is_stacked else _MAG_DR[0]), y_max)  # in dB
        ax.grid()
        ax.set_xlabel("Frequency (Hz)" if ch >= n_ch - 1 else "")
        ax.set_ylabel("PSD Magnitude (dB)")

        if not is_stacked:
            # Spectrogram
            ax = axes[ch, 2]
            [specs, _, _, img] = ax.specgram(
                x=data[ch_plot, :].T,
                Fs=fs,  # in Hz
                NFFT=1024,  # in samples
                noverlap=512,  # in samples
                mode="magnitude",
                scale="dB",
            )
            specs_peak = np.ceil(
                20 * np.log10(np.max(np.abs(specs))) / 5) * 5  # in dB
            ax.set_yscale("log")
            ax.set_ylim(100, fs / 2)  # in Hz
            ax.set_xlabel("Time (s)" if ch >= n_ch - 1 else "")
            ax.set_ylabel("Frequency (Hz)")
            plt.colorbar(mappable=img, ax=ax, label="Magnitude (dB)")
            img.set_clim(specs_peak - _MAG_DR[1], specs_peak)  # in dB

    plt.show()
