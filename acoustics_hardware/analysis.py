import numpy as np
import scipy.signal
from . import signal_tools


def dB(x, power=False, safe_zeros=True):
    if safe_zeros:
        x = signal_tools.nonzero_signals(x)
    if power:
        if np.any(x < 0):
            return 10 * np.log10(x + 0j)
        return 10 * np.log10(x)
    else:
        return 20 * np.log10(np.abs(x))


def sweep_deconvolution(
    reference_signal,
    measured_signal,
    samplerate=None,
    lower_cutoff=None,
    upper_cutoff=None,
    truncation_length=None,
    output_length=None,
    fade_in=None,
    fade_out=None,
    ir_orders=1,
    inversion_method='regularized',
    **kwargs
):
    if isinstance(inversion_method, str):
        if inversion_method.lower() == 'filtering':
            inversion_method = _filtered_deconvolution
        elif inversion_method.lower() == 'regularized':
            inversion_method = _regularized_deconvolution
        elif inversion_method.lower() == 'analytic':
            inversion_method = _exponential_sweep_analytical_deconvolution
        elif inversion_method.lower() == 'time-reversal':
            inversion_method = _exponential_sweep_time_reversal_deconvolution
        else:
            raise ValueError(f'Unknown inverse filtering method {inversion_method}')

    impulse_response = inversion_method(
        reference_signal=reference_signal,
        measured_signal=measured_signal,
        samplerate=samplerate,
        lower_cutoff=lower_cutoff,
        upper_cutoff=upper_cutoff,
        **kwargs
    )

    if ir_orders == 'whole':
        return impulse_response

    num_samples = impulse_response.shape[-1] // 2

    irs = [impulse_response[..., :num_samples]]
    if ir_orders > 1:
        phase_rate = reference_signal.size / np.log(upper_cutoff / lower_cutoff)
        lags = np.log(np.arange(1, ir_orders + 1)) * phase_rate  # How much each impulse response is before the linear one
        indices = num_samples - np.ceil(lags).astype(int)  # The index where each of the impulse responses start
        for order in range(1, ir_orders):  # Starts with order 1, corresponding to the first harmonic.
            irs.append(impulse_response[indices[order]:indices[order - 1]])

    irs = signal_tools.truncate_signals(irs, length=truncation_length, samplerate=samplerate)
    irs = signal_tools.extend_signals(irs, length=output_length, samplerate=samplerate)
    irs = signal_tools.fade_signals(irs, fade_in=fade_in, fade_out=fade_out, samplerate=samplerate)

    if len(irs) == 1:
        return irs[0]
    return irs


def _filtered_deconvolution(
    reference_signal,
    measured_signal,
    samplerate=None,
    lower_cutoff=None,
    upper_cutoff=None,
    **kwargs
):
    measured_signal = np.asarray(measured_signal)
    num_samples = measured_signal.shape[-1]
    impulse_response = np.fft.irfft(
        np.fft.rfft(measured_signal, 2 * num_samples)
        / np.fft.rfft(reference_signal, 2 * num_samples)
    )

    filter_args = kwargs or {}
    if lower_cutoff is not None and upper_cutoff is not None:
        filter_args.setdefault('Wn', (lower_cutoff, upper_cutoff))
        filter_args['btype'] = 'bandpass'
    elif lower_cutoff is not None:
        filter_args.setdefault('Wn', lower_cutoff)
        filter_args['btype'] = 'highpass'
    elif upper_cutoff is not None:
        filter_args.setdefault('Wn', upper_cutoff)
        filter_args['btype'] = 'lowpass'
    else:
        return impulse_response

    filter_args.setdefault('N', 8)
    filter_args.setdefault('fs', samplerate)
    filter_args.setdefault('ftype', 'butter')
    filter_args['output'] = 'sos'

    sos = scipy.signal.iirfilter(**filter_args)
    impulse_response = scipy.signal.sosfilt(sos, impulse_response)
    return impulse_response


def _regularized_deconvolution(
    reference_signal,
    measured_signal,
    samplerate=None,
    lower_cutoff=None,
    upper_cutoff=None,
    transition_width=1 / 12,
    interior_regularization=0.01,
    exterior_regularization=1,
):
    measured_signal = np.asarray(measured_signal)
    num_samples = measured_signal.shape[-1]

    reference_spectrum = np.fft.rfft(reference_signal, 2 * num_samples)
    samplerate = samplerate or 2  # No given samplerate -> frequencies normalized to Nyquist as in iirfilter
    f = np.fft.rfftfreq(2 * num_samples, 1 / samplerate)
    if lower_cutoff is None:
        lower_cutoff = f[1]
    if upper_cutoff is None:
        upper_cutoff = f[-1]
    lower_cutoff_interior = lower_cutoff * 2**transition_width
    upper_cutoff_interior = upper_cutoff / 2**transition_width

    regularization = np.zeros(reference_spectrum.shape)
    interior_idx = (lower_cutoff_interior <= f) & (f <= upper_cutoff_interior)
    regularization[interior_idx] = 1 / f[interior_idx] * interior_regularization
    regularization[0] = 0

    regularization[f <= lower_cutoff] = exterior_regularization / lower_cutoff
    slope_idx = (lower_cutoff < f) & (f < lower_cutoff_interior)
    regularization[slope_idx] = np.geomspace(exterior_regularization / lower_cutoff, interior_regularization / lower_cutoff_interior, np.sum(slope_idx))

    slope_idx = (upper_cutoff_interior < f) & (f < upper_cutoff)
    regularization[upper_cutoff <= f] = exterior_regularization / upper_cutoff
    regularization[slope_idx] = np.geomspace(interior_regularization / upper_cutoff_interior, exterior_regularization / upper_cutoff, np.sum(slope_idx))

    regularization *= np.mean(np.abs(reference_spectrum[interior_idx])**2 / regularization[interior_idx]) * interior_regularization
    inverse_spectrum = reference_spectrum.conj() / (reference_spectrum.conj() * reference_spectrum + regularization)

    impulse_response = np.fft.irfft(np.fft.rfft(measured_signal, n=2 * num_samples) * inverse_spectrum)
    return impulse_response


def _exponential_sweep_analytical_deconvolution(
    reference_signal,
    measured_signal,
    samplerate,
    lower_cutoff,
    upper_cutoff,
):
    num_samples = reference_signal.size
    phase_rate = num_samples / samplerate / np.log(upper_cutoff / lower_cutoff)
    f = np.fft.rfftfreq(num_samples, 1 / samplerate)
    with np.errstate(divide='ignore', invalid='ignore'):
        inverse_spectrum = 2 * (f / phase_rate)**0.5 * np.exp(-2j * np.pi * f * phase_rate * (1 - np.log(f / lower_cutoff)) + 1j * np.pi / 4)
    inverse_spectrum[0] = 0
    # We have to go back to time to zero pad.
    # It's much easier to generate the correct spectrum with the original reference length.
    inverse_filter = np.fft.irfft(inverse_spectrum) / samplerate

    measured_signal = np.asarray(measured_signal)
    num_samples = measured_signal.shape[-1]

    impulse_response = np.fft.irfft(
        np.fft.rfft(measured_signal, n=2 * num_samples)
        * np.fft.rfft(inverse_filter, n=2 * num_samples),
    )
    impulse_response = np.roll(impulse_response, -reference_signal.size, axis=-1)

    return impulse_response


def _exponential_sweep_time_reversal_deconvolution(
    reference_signal,
    measured_signal,
    lower_cutoff,
    upper_cutoff,
    samplerate=None,
):
    amplitude_correction = 10**(np.linspace(0, -6 * np.log2(upper_cutoff / lower_cutoff), reference_signal.size) / 20)
    inverse_filter = reference_signal[-1::-1] * amplitude_correction

    measured_signal = np.asarray(measured_signal)
    num_samples = measured_signal.shape[-1]

    impulse_response = np.fft.irfft(
        np.fft.rfft(measured_signal, n=2 * num_samples, axis=-1)
        * np.fft.rfft(inverse_filter, n=2 * num_samples),
        axis=-1
    )
    impulse_response = np.roll(impulse_response, -reference_signal.size, axis=-1)

    return impulse_response


def mls_analysis(reference, output):
    output = np.atleast_2d(output)
    seq_len = len(reference)
    order = np.round(np.log2(seq_len + 1)).astype(int)
    reps = int(output.shape[1] / seq_len)

    if reps > 1:
        response = output[:, seq_len:reps * seq_len].reshape((-1, reps - 1, seq_len)).mean(axis=1)
    else:
        response = output[:, :seq_len]

    ps = np.zeros(seq_len, dtype=np.int64)
    for idx in range(seq_len):
        for s in range(order):
            ps[idx] += reference[(idx - s) % seq_len] << (order - 1 - s)

    indices = np.argsort(ps)[2**np.arange(order - 1, -1, -1) - 1]
    pl = np.zeros(seq_len, dtype=np.int64)
    for idx in range(seq_len):
        for s in range(order):
            pl[idx] += reference[(indices[s] - idx) % seq_len] << (order - s - 1)

    transform = np.zeros((output.shape[0], seq_len + 1))
    transform[:, ps] = response
    for _ in range(order):
        transform = np.concatenate((transform[:, ::2] + transform[:, 1::2], transform[:, ::2] - transform[:, 1::2]), axis=1)
    ir = transform[:, pl] / (seq_len + 1)
    return np.squeeze(ir)
