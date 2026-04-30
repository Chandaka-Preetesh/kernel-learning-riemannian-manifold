"""
kernel_learning.py
------------------
Construct the initial kernel matrix.

From the paper:
    K0  =  X  A  X^T

where X ∈ R^{N×d} are the vectorised descriptors and A ∈ R^{d×d} is the
learned Mahalanobis matrix.

This module also provides a Gaussian (RBF) kernel baseline for comparison.
"""

import numpy as np


def initial_kernel(X: np.ndarray, A: np.ndarray) -> np.ndarray:
    """
    Compute the initial kernel matrix  K0 = X A X^T.

    Parameters
    ----------
    X : np.ndarray  shape (N, d)
    A : np.ndarray  shape (d, d)   SPD Mahalanobis matrix.

    Returns
    -------
    K0 : np.ndarray  shape (N, N)
    """
    # X A X^T   — compute X A first, then multiply by X^T
    K0 = (X @ A) @ X.T          # (N, N)

    # Symmetrise (numerics)
    K0 = (K0 + K0.T) / 2.0

    # Project onto PSD cone (clip negative eigenvalues)
    eigvals, eigvecs = np.linalg.eigh(K0)
    eigvals = np.clip(eigvals, 0.0, None)
    K0 = eigvecs @ np.diag(eigvals) @ eigvecs.T
    K0 = (K0 + K0.T) / 2.0   # re-symmetrise

    return K0


def gaussian_kernel(X: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Standard Gaussian (RBF) kernel for comparison.

    K_ij = exp( -||xi - xj||^2 / (2 σ^2) )

    Parameters
    ----------
    X : np.ndarray  shape (N, d)
    sigma : float   Bandwidth parameter.

    Returns
    -------
    K : np.ndarray  shape (N, N)
    """
    # Squared Euclidean distance matrix
    sq_norms = np.sum(X ** 2, axis=1)                   # (N,)
    D2 = sq_norms[:, None] + sq_norms[None, :] - 2 * X @ X.T

    # Clip negative values from numerical noise
    D2 = np.clip(D2, 0, None)

    K = np.exp(-D2 / (2.0 * sigma ** 2))
    return K


def log_euclidean_kernel(cov_list: list, sigma: float = 1.0) -> np.ndarray:
    """
    Log-Euclidean kernel: K_ij = exp(-||logm(Ci) - logm(Cj)||^2 / (2σ^2))

    Parameters
    ----------
    cov_list : list[np.ndarray]   List of SPD matrices, each (d, d).
    sigma : float   Bandwidth.

    Returns
    -------
    K : np.ndarray  shape (N, N)
    """
    from log_mapping import log_map

    logs = [log_map(C).flatten() for C in cov_list]
    X_log = np.array(logs)   # (N, d*d)
    return gaussian_kernel(X_log, sigma=sigma)
