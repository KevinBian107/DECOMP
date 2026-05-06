"""Lag analysis and peri-event averaging on residual canonical variates.

Given the leading cross-region canonical variates U_A(t) and U_B(t) extracted under a given
Z (typically Z_rich), this module asks two questions that CCA itself does not optimise for:

1. Does U_A(t) systematically lead/lag U_B(t)?
   Cross-correlation function corr(U_A(t), U_B(t + tau)) over a window of lags.
   - Peak at tau = 0, symmetric: shared upstream drive.
   - Peak at tau != 0: real cortico-cortical / cortico-cerebellar coupling.

2. Do U_A(t), U_B(t) carry task-related structure?
   Peri-event-time averages aligned to stimulus onset, first movement, feedback.
   - Systematic peri-event deflection: residual carries task-related signal.
   - Flat peri-event average: residual is task-independent (global state or artefact).

Both analyses use the leading canonical component (k = 1) by default — the strongest axis
of cross-region coupling, where the lag and event-locking signatures should be most visible.
"""

from __future__ import annotations

import numpy as np


def cross_correlation_function(U_A: np.ndarray, U_B: np.ndarray,
                                 max_lag_bins: int = 25) -> tuple[np.ndarray, np.ndarray]:
    """corr(U_A(t), U_B(t + tau)) for tau in [-max_lag_bins, +max_lag_bins].

    Positive tau → U_A leads U_B (a peak at +tau means U_A is informative about U_B's
    future). NaN-tolerant via masked Pearson.
    """
    a = np.asarray(U_A, dtype=float)
    b = np.asarray(U_B, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    a, b = a[valid], b[valid]
    if a.std() == 0 or b.std() == 0:
        lags = np.arange(-max_lag_bins, max_lag_bins + 1)
        return lags, np.zeros(len(lags))
    a = (a - a.mean()) / a.std()
    b = (b - b.mean()) / b.std()
    T = len(a)
    lags = np.arange(-max_lag_bins, max_lag_bins + 1)
    ccf = np.zeros(len(lags))
    for i, tau in enumerate(lags):
        if tau >= 0:
            ccf[i] = float(np.dot(a[:T - tau], b[tau:]) / (T - tau))
        else:
            ccf[i] = float(np.dot(a[-tau:], b[:T + tau]) / (T + tau))
    return lags, ccf


def _phase_shuffle(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(x)
    xf = np.fft.rfft(np.nan_to_num(x))
    phases = np.exp(1j * rng.uniform(0, 2 * np.pi, size=xf.shape))
    phases[0] = 1.0
    if n % 2 == 0:
        phases[-1] = 1.0
    return np.fft.irfft(xf * phases, n=n)


def ccf_phase_null(U_A: np.ndarray, U_B: np.ndarray, max_lag_bins: int = 25,
                    n_surrogates: int = 100, random_state: int = 0) -> np.ndarray:
    """Phase-randomise U_A independently `n_surrogates` times, recompute CCF each time.

    Returns array of shape (n_surrogates, n_lags). The 99% percentile band of the absolute
    value gives a per-lag null cutoff.
    """
    rng = np.random.default_rng(random_state)
    nulls = []
    for _ in range(n_surrogates):
        a_shuf = _phase_shuffle(U_A, rng)
        _, ccf = cross_correlation_function(a_shuf, U_B, max_lag_bins=max_lag_bins)
        nulls.append(ccf)
    return np.array(nulls)


def peri_event_average(U: np.ndarray, event_times: np.ndarray, bin_centers: np.ndarray,
                        window_s: float = 2.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Peri-event-time average of U(t) within ±window_s of each event.

    Returns (t_axis, mean, sem, n_events_used). Events outside the recording window or
    landing on NaN regions are skipped.
    """
    bin_s = float(np.median(np.diff(bin_centers)))
    half_n = int(round(window_s / bin_s))
    t_axis = np.arange(-half_n, half_n + 1) * bin_s

    et = np.asarray(event_times, dtype=float)
    et = et[np.isfinite(et)]
    aligned = []
    T = len(U)
    for e in et:
        if e < bin_centers[0] or e > bin_centers[-1]:
            continue
        idx = int(np.searchsorted(bin_centers, e))
        i0, i1 = idx - half_n, idx + half_n + 1
        if i0 < 0 or i1 > T:
            continue
        snippet = U[i0:i1]
        if not np.all(np.isfinite(snippet)):
            continue
        aligned.append(snippet)
    if not aligned:
        return t_axis, np.full_like(t_axis, np.nan), np.full_like(t_axis, np.nan), 0
    A = np.array(aligned)
    n = A.shape[0]
    return t_axis, A.mean(axis=0), A.std(axis=0, ddof=1) / np.sqrt(n), n
