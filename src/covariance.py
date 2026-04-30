"""
covariance.py
-------------
Compute covariance (SPD) matrix from a feature matrix.

From the paper Eq.(2)(3):
    C = (1/n) * (Z - μ)^T (Z - μ)

A small ε is added to the diagonal to guarantee Symmetric Positive Definiteness
(SPD), which is required for the matrix logarithm to be well-defined.
"""

import numpy as np


def compute_covariance(features: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Parameters
    ----------
    features : np.ndarray  shape (n, d)
        Feature matrix (e.g. from one image region).
    eps : float
        Small regulariser added to diagonal for SPD guarantee.

    Returns
    -------
    C : np.ndarray  shape (d, d)
        Symmetric positive-definite covariance matrix.
    """
    n, d = features.shape

    # Mean-centre the features
    mean = np.mean(features, axis=0)          # (d,)
    centred = features - mean                  # (n, d)

    # Sample covariance
    C = centred.T @ centred / n                # (d, d)

    # Regularise: ensure SPD
    C += eps * np.eye(d)

    return C
