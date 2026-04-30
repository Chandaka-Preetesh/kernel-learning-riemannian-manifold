"""
log_mapping.py
--------------
Log-Euclidean mapping and geodesic distance.

The log-Euclidean framework maps SPD matrices into a flat (Euclidean) tangent
space via the matrix logarithm:

    S  =  logm(C)          (d × d  symmetric matrix)

Geodesic distance under the log-Euclidean metric:

    d(C1, C2)  =  || logm(C1) - logm(C2) ||_F
"""

import numpy as np
from scipy.linalg import logm


# ---------- mapping ----------

def log_map(C: np.ndarray) -> np.ndarray:
    """
    Compute the matrix logarithm of an SPD matrix.

    Parameters
    ----------
    C : np.ndarray  shape (d, d)   SPD matrix.

    Returns
    -------
    S : np.ndarray  shape (d, d)   Symmetric matrix in tangent space.
    """
    S = logm(C)

    # logm can introduce tiny imaginary parts due to numerics — discard them
    S = np.real(S)

    # Re-symmetrise (logm of SPD is symmetric in theory)
    S = (S + S.T) / 2.0

    return S


# ---------- vectorisation ----------

def vectorize(S: np.ndarray) -> np.ndarray:
    """
    Flatten a symmetric matrix into a 1-D vector.

    Parameters
    ----------
    S : np.ndarray  shape (d, d)

    Returns
    -------
    v : np.ndarray  shape (d*d,)
    """
    return S.flatten()


# ---------- geodesic distance ----------

def geodesic_distance(C1: np.ndarray, C2: np.ndarray) -> float:
    """
    Log-Euclidean geodesic distance between two SPD matrices.

    d(C1, C2) = || logm(C1) - logm(C2) ||_F

    Parameters
    ----------
    C1, C2 : np.ndarray  shape (d, d)   SPD matrices.

    Returns
    -------
    dist : float   Non-negative scalar.
    """
    return float(np.linalg.norm(log_map(C1) - log_map(C2), ord='fro'))


def geodesic_distance_matrix(cov_list: list) -> np.ndarray:
    """
    Build the full N×N geodesic distance matrix for a list of SPD matrices.

    Parameters
    ----------
    cov_list : list[np.ndarray]
        List of N covariance matrices, each (d, d).

    Returns
    -------
    D : np.ndarray  shape (N, N)
        Pairwise geodesic distance matrix (symmetric, zero diagonal).
    """
    N = len(cov_list)

    # Pre-compute all log maps to avoid redundant work
    logs = [log_map(C) for C in cov_list]

    D = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            d = float(np.linalg.norm(logs[i] - logs[j], ord='fro'))
            D[i, j] = d
            D[j, i] = d

    return D
