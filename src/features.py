"""
features.py
-----------
Extract per-pixel features from a grayscale image.

Features per pixel (d=7):
    [x, y, I, Ix, Iy, Ixx, Iyy]

where:
    x, y   = spatial coordinates (normalised to [0,1])
    I      = pixel intensity (normalised to [0,1])
    Ix, Iy = first-order image gradients  (Sobel)
    Ixx, Iyy = second-order derivatives   (Sobel on gradients)
"""

import numpy as np
import cv2


def extract_features(img: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------
    img : np.ndarray
        Grayscale image of shape (H, W), dtype uint8 or float.

    Returns
    -------
    features : np.ndarray  shape (H*W, 7)
        Feature matrix Z  ∈  R^{n x d}  with d=7.
    """
    # Ensure float64 for stable gradient computation
    if img.dtype != np.float64:
        img = img.astype(np.float64) / 255.0

    H, W = img.shape

    # --- Spatial coordinates (normalised) ---
    yy, xx = np.mgrid[0:H, 0:W]
    x_norm = xx.astype(np.float64) / max(W - 1, 1)
    y_norm = yy.astype(np.float64) / max(H - 1, 1)

    # --- Intensity (already [0,1]) ---
    I = img

    # --- First-order gradients (Sobel 3×3) ---
    Ix = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    # --- Second-order derivatives ---
    Ixx = cv2.Sobel(Ix, cv2.CV_64F, 1, 0, ksize=3)
    Iyy = cv2.Sobel(Iy, cv2.CV_64F, 0, 1, ksize=3)

    # Stack all features and reshape to (n, d)
    features = np.stack([
        x_norm.ravel(),
        y_norm.ravel(),
        I.ravel(),
        Ix.ravel(),
        Iy.ravel(),
        Ixx.ravel(),
        Iyy.ravel()
    ], axis=-1)  # (H*W, 7)

    return features
