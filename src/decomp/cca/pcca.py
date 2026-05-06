"""Partial CCA via residualize-then-CCA, using a fast SVD-based CCA solver.

`cca-zoo>=3.0.0` (March 2026 rewrite) dropped its `PartialCCA` class, and `scikit-learn`
ships only NIPALS-based CCA which is prohibitively slow on long timeseries (T~10^5+).

Math (Hardoon, Szedmak & Shawe-Taylor 2004):
    Standardize X, Y.
    C_xx = X^T X / T,   C_yy = Y^T Y / T,   C_xy = X^T Y / T
    A = C_xx^{-1/2} @ C_xy @ C_yy^{-1/2}
    U, sigma, V^T = SVD(A)            # singular values are canonical correlations

For partial CCA, residualize X and Y on Z first (Frisch-Waugh-Lovell):
    X_r = X - Z @ pinv(Z) @ X
    Y_r = Y - Z @ pinv(Z) @ Y
then compute CCA(X_r, Y_r).
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

_REG = 1e-6


def _whiten(C: np.ndarray, reg: float = _REG) -> np.ndarray:
    """Symmetric inverse square root via eigen-decomposition + Tikhonov."""
    w, V = np.linalg.eigh(C)
    w = np.clip(w, reg, None)
    return V @ np.diag(1.0 / np.sqrt(w)) @ V.T


def cca_svd(X: np.ndarray, Y: np.ndarray, n_components: int | None = None,
            reg: float = _REG):
    """Fast SVD-based CCA. Returns (rho, A, B) with rho the canonical correlations."""
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    T = X.shape[0]
    Cxx = Xc.T @ Xc / T + reg * np.eye(X.shape[1])
    Cyy = Yc.T @ Yc / T + reg * np.eye(Y.shape[1])
    Cxy = Xc.T @ Yc / T
    Wx = _whiten(Cxx, reg)
    Wy = _whiten(Cyy, reg)
    A = Wx @ Cxy @ Wy
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    rho = np.clip(S, 0.0, 1.0)
    if n_components is not None:
        rho = rho[:n_components]
        U = U[:, :n_components]
        Vt = Vt[:n_components, :]
    A_x = Wx @ U                  # (d_x, k) — projection for X
    B_y = Wy @ Vt.T               # (d_y, k) — projection for Y
    return rho, A_x, B_y


def residualize(Y: np.ndarray, Z: np.ndarray | None) -> np.ndarray:
    """Linear-regress out Z from Y (column-wise). Returns Y unchanged when Z is None."""
    if Z is None:
        return Y
    return Y - LinearRegression(fit_intercept=True).fit(Z, Y).predict(Z)


def fit_pcca(X: np.ndarray, Y: np.ndarray, Z: np.ndarray | None, n_components: int):
    """Fit a partial CCA. When Z is None this is identical to vanilla CCA."""
    return cca_svd(residualize(X, Z), residualize(Y, Z), n_components=n_components)


def cv_residual_variates(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray | None = None,
    n_components: int = 8,
    n_splits: int = 5,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract cross-validated (partial) canonical variates U_A(t), U_B(t) of length T.

    For each fold:
      - Fit residualisation Z->X and Z->Y on train, apply to test (Frisch-Waugh-Lovell).
      - Fit CCA loadings on residualised train via SVD.
      - Project residualised test data through trained loadings, store at the test indices.

    Sign alignment: within each fold, flip B_y per-component so train (U_A, U_B) is
    positively correlated. Across folds, flip A_x and B_y to align with fold-1 loadings,
    preserving canonical-direction sign so test-fold projections concatenate cleanly.

    Returns: (U_A, U_B) each of shape (T, n_comp).
    """
    T = X.shape[0]
    n_comp = min(n_components, X.shape[1], Y.shape[1])
    U_A = np.full((T, n_comp), np.nan)
    U_B = np.full((T, n_comp), np.nan)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    A_ref: np.ndarray | None = None

    for tr_idx, te_idx in kf.split(X):
        Xtr, Xte = X[tr_idx], X[te_idx]
        Ytr, Yte = Y[tr_idx], Y[te_idx]
        if Z is not None:
            Ztr, Zte = Z[tr_idx], Z[te_idx]
            rx = LinearRegression(fit_intercept=True).fit(Ztr, Xtr)
            ry = LinearRegression(fit_intercept=True).fit(Ztr, Ytr)
            Xtr_r, Xte_r = Xtr - rx.predict(Ztr), Xte - rx.predict(Zte)
            Ytr_r, Yte_r = Ytr - ry.predict(Ztr), Yte - ry.predict(Zte)
        else:
            Xtr_r, Xte_r, Ytr_r, Yte_r = Xtr, Xte, Ytr, Yte

        _, A_x, B_y = cca_svd(Xtr_r, Ytr_r, n_components=n_comp)

        # Within-fold sign: ensure train (U_A, U_B) per-component correlation is positive.
        Xs_tr = (Xtr_r - Xtr_r.mean(axis=0, keepdims=True)) @ A_x
        Ys_tr = (Ytr_r - Ytr_r.mean(axis=0, keepdims=True)) @ B_y
        for k in range(n_comp):
            if Xs_tr[:, k].std() > 0 and Ys_tr[:, k].std() > 0:
                if np.corrcoef(Xs_tr[:, k], Ys_tr[:, k])[0, 1] < 0:
                    B_y[:, k] *= -1
        # Cross-fold sign: align A_x with fold-1; flip B_y to preserve correlation sign.
        if A_ref is None:
            A_ref = A_x.copy()
        else:
            for k in range(n_comp):
                if np.dot(A_x[:, k], A_ref[:, k]) < 0:
                    A_x[:, k] *= -1
                    B_y[:, k] *= -1

        Xc = Xte_r - Xte_r.mean(axis=0, keepdims=True)
        Yc = Yte_r - Yte_r.mean(axis=0, keepdims=True)
        U_A[te_idx] = Xc @ A_x
        U_B[te_idx] = Yc @ B_y

    return U_A, U_B


def cv_canonical_correlations(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray | None = None,
    n_components: int = 10,
    n_splits: int = 5,
    random_state: int = 0,
) -> dict:
    """Cross-validated canonical correlations (or partial canonical correlations).

    For each fold:
      - Fit residualization (Z->X and Z->Y) on train, apply to test.
      - Fit CCA on residualized train via SVD; project residualized test; per-component rho
        is the Pearson correlation between projected scores on test fold.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rhos = []
    for tr_idx, te_idx in kf.split(X):
        Xtr, Xte = X[tr_idx], X[te_idx]
        Ytr, Yte = Y[tr_idx], Y[te_idx]
        if Z is not None:
            Ztr, Zte = Z[tr_idx], Z[te_idx]
            rx = LinearRegression(fit_intercept=True).fit(Ztr, Xtr)
            ry = LinearRegression(fit_intercept=True).fit(Ztr, Ytr)
            Xtr_r, Xte_r = Xtr - rx.predict(Ztr), Xte - rx.predict(Zte)
            Ytr_r, Yte_r = Ytr - ry.predict(Ztr), Yte - ry.predict(Zte)
        else:
            Xtr_r, Xte_r, Ytr_r, Yte_r = Xtr, Xte, Ytr, Yte

        n_comp = min(n_components, Xtr_r.shape[1], Ytr_r.shape[1])
        _, A_x, B_y = cca_svd(Xtr_r, Ytr_r, n_components=n_comp)
        # project test fold and compute per-component test-set correlations
        Xc = Xte_r - Xte_r.mean(axis=0, keepdims=True)
        Yc = Yte_r - Yte_r.mean(axis=0, keepdims=True)
        Xs = Xc @ A_x
        Ys = Yc @ B_y
        rho_te = np.array([
            np.corrcoef(Xs[:, k], Ys[:, k])[0, 1] if Xs[:, k].std() > 0 else 0.0
            for k in range(n_comp)
        ])
        # pad to n_components if smaller
        if len(rho_te) < n_components:
            rho_te = np.concatenate([rho_te, np.zeros(n_components - len(rho_te))])
        rhos.append(rho_te[:n_components])

    rhos = np.array(rhos)
    return {
        "rho_per_fold": rhos,
        "rho_mean": rhos.mean(axis=0),
        "rho_se": rhos.std(axis=0, ddof=1) / np.sqrt(n_splits) if n_splits > 1 else np.zeros(n_components),
    }
