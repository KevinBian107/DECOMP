"""Design matrix construction for the per-neuron encoding GLM.

Builds raised-cosine kernel bases over a 20 ms grid and stacks them into a single design
matrix. Math is identical to `neurencoding`'s `nonlinear_rcos` + `DesignMatrix.compile`,
but written directly in numpy so the module is independent of a specific upstream API.

Kernel parameters mirror `paper-brain-wide-map/brainwidemap/encoding/pipelines/02_fit_sessions.py:99-124`:
  stimL/stimR : 0.4 s post-stim, 5 raised-cosine bases
  fmoveL/fmoveR : ±0.2 s around first-movement, 5 bases
  correct/incorrect : feedback kernels
  wheel : continuous, 0.4 s lookback, 5 bases
  motion energy (whisker, body) : continuous, 0.4 s lookback, 5 bases
  pupil : continuous, 0.4 s lookback, 3 bases
  prior : block-probability step regressor (1 column)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..data.binning import BIN_S, BinnedSession


@dataclass
class DesignSpec:
    """Layout of the (T, p) design matrix: which columns belong to which kernel group."""
    columns: list[str]
    groups: dict[str, list[int]]  # group_name -> list of column indices


def raised_cosine_basis(n_bins: int, n_basis: int, lin_offset: float = 1e-3) -> np.ndarray:
    """Pillow-style log-spaced raised-cosine basis. Returns (n_bins, n_basis) array."""
    if n_basis < 1 or n_bins < 1:
        return np.zeros((max(n_bins, 0), max(n_basis, 0)))
    t = np.arange(n_bins)
    nl = np.log(t + lin_offset + 1e-12)
    nl_min, nl_max = nl.min(), nl.max()
    centers = np.linspace(nl_min, nl_max, n_basis)
    width = (centers[1] - centers[0]) if n_basis > 1 else (nl_max - nl_min)
    basis = np.zeros((n_bins, n_basis))
    for k, c in enumerate(centers):
        u = (nl - c) * np.pi / (2 * width)
        u = np.clip(u, -np.pi, np.pi)
        basis[:, k] = (np.cos(u) + 1) / 2
    # Normalize columns to unit max for interpretability
    maxes = basis.max(axis=0)
    maxes[maxes == 0] = 1.0
    basis /= maxes
    return basis


def _convolve_event_train(event_bins: np.ndarray, basis: np.ndarray, n_total: int) -> np.ndarray:
    """Convolve a binary event train (n_total,) with a basis (n_bins, n_basis).

    Returns (n_total, n_basis): each column is the train convolved with one basis kernel,
    truncated to n_total.
    """
    n_basis = basis.shape[1]
    out = np.zeros((n_total, n_basis))
    for k in range(n_basis):
        kern = basis[:, k]
        out[:, k] = np.convolve(event_bins, kern, mode="full")[:n_total]
    return out


def _convolve_continuous(x: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Convolve a continuous regressor x (T,) with each basis kernel.

    The basis represents the post-event impulse response; we convolve x with the basis to
    get T x n_basis lagged versions of x.
    """
    return _convolve_event_train(x, basis, n_total=len(x))


def _bin_event_times(event_times: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """Histogram event timestamps into a binary bin train."""
    if event_times is None or len(event_times) == 0:
        return np.zeros(len(bin_edges) - 1, dtype=np.float64)
    counts, _ = np.histogram(np.asarray(event_times, dtype=float), bins=bin_edges)
    return counts.astype(np.float64)


def build_design_matrix(binned: BinnedSession, trials: pd.DataFrame) -> tuple[np.ndarray, DesignSpec]:
    """Build (T, p) design matrix from a binned session + trials table."""
    T = len(binned.bin_centers)
    edges = binned.bin_edges
    columns: list[str] = []
    blocks: list[np.ndarray] = []
    groups: dict[str, list[int]] = {}

    def add(group: str, name: str, mat: np.ndarray) -> None:
        if mat.ndim == 1:
            mat = mat[:, None]
        groups.setdefault(group, [])
        for k in range(mat.shape[1]):
            groups[group].append(len(columns))
            columns.append(f"{name}_{k}" if mat.shape[1] > 1 else name)
        blocks.append(mat)

    # ---- Bases --------------------------------------------------------------
    n_bins_400ms = int(round(0.4 / binned.bin_s))
    n_bins_200ms = int(round(0.2 / binned.bin_s))
    basis_post = raised_cosine_basis(n_bins_400ms, n_basis=5)
    basis_around = raised_cosine_basis(n_bins_200ms, n_basis=5)

    # ---- Event regressors ---------------------------------------------------
    if trials is not None and len(trials):
        # Stimulus
        stimL = trials.loc[trials["contrastLeft"].notna(), "stimOn_times"].to_numpy()
        stimR = trials.loc[trials["contrastRight"].notna(), "stimOn_times"].to_numpy()
        add("stim", "stimL", _convolve_event_train(_bin_event_times(stimL, edges), basis_post, T))
        add("stim", "stimR", _convolve_event_train(_bin_event_times(stimR, edges), basis_post, T))

        # First-movement (centered: bin around)
        if "firstMovement_times" in trials.columns:
            fmove = trials["firstMovement_times"].dropna().to_numpy()
            add("movement", "fmove", _convolve_event_train(_bin_event_times(fmove, edges),
                                                            basis_around, T))

        # Feedback
        if "feedbackType" in trials.columns and "feedback_times" in trials.columns:
            corr = trials.loc[trials["feedbackType"] == 1, "feedback_times"].dropna().to_numpy()
            err = trials.loc[trials["feedbackType"] == -1, "feedback_times"].dropna().to_numpy()
            add("feedback", "feedback_correct",
                _convolve_event_train(_bin_event_times(corr, edges), basis_post, T))
            add("feedback", "feedback_error",
                _convolve_event_train(_bin_event_times(err, edges), basis_post, T))

        # Choice (encode as event at response_times signed by choice)
        if "choice" in trials.columns and "response_times" in trials.columns:
            tt = trials[["response_times", "choice"]].dropna()
            chL = tt.loc[tt["choice"] == 1, "response_times"].to_numpy()
            chR = tt.loc[tt["choice"] == -1, "response_times"].to_numpy()
            add("choice", "choice_left",
                _convolve_event_train(_bin_event_times(chL, edges), basis_post, T))
            add("choice", "choice_right",
                _convolve_event_train(_bin_event_times(chR, edges), basis_post, T))

        # Block prior (step function)
        if "probabilityLeft" in trials.columns and "stimOn_times" in trials.columns:
            prior = np.zeros(T)
            tt = trials[["stimOn_times", "probabilityLeft"]].dropna().sort_values("stimOn_times")
            for ts, p in zip(tt["stimOn_times"].to_numpy(), tt["probabilityLeft"].to_numpy()):
                idx = np.searchsorted(binned.bin_centers, ts)
                prior[idx:] = p
            add("prior", "prior", prior)

    # ---- Continuous regressors ---------------------------------------------
    cov = binned.covariates
    if "wheel_velocity" in cov.columns:
        add("wheel", "wheel_v", _convolve_continuous(cov["wheel_velocity"].to_numpy(), basis_post))
    if "wheel_acceleration" in cov.columns:
        add("wheel", "wheel_a", _convolve_continuous(cov["wheel_acceleration"].to_numpy(), basis_post))
    if "me_whisker" in cov.columns:
        add("me_face", "me_whisker", _convolve_continuous(cov["me_whisker"].to_numpy(), basis_post))
    if "me_body" in cov.columns:
        add("me_body", "me_body", _convolve_continuous(cov["me_body"].to_numpy(), basis_post))
    if "pupil" in cov.columns:
        basis_pupil = raised_cosine_basis(n_bins_400ms, n_basis=3)
        add("pupil", "pupil", _convolve_continuous(cov["pupil"].to_numpy(), basis_pupil))
    if "lick_rate" in cov.columns:
        add("movement", "lick", _convolve_continuous(cov["lick_rate"].to_numpy(), basis_post))

    if not blocks:
        raise RuntimeError("Design matrix is empty -- no covariates found in binned session.")

    X = np.concatenate(blocks, axis=1)
    return X, DesignSpec(columns=columns, groups=groups)


# Aggregate "movement" group includes wheel, motion energy, fmove, lick.
MOVEMENT_GROUPS = ("wheel", "movement", "me_face", "me_body")
STIMULUS_GROUPS = ("stim",)
CHOICE_GROUPS = ("choice",)
