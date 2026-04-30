"""
main.py
-------
Full end-to-end pipeline for:

    "Kernel Learning on Riemannian Manifold with Geodesic Distance Preservation"

Pipeline steps:
    1.  Load / generate images
    2.  Extract features  (x, y, I, Ix, Iy, Ixx, Iyy)
    3.  Compute covariance matrices  (SPD)
    4.  Log-Euclidean mapping  (logm)
    5.  Vectorise
    6.  Compute geodesic distance matrix
    7.  Learn Mahalanobis matrix  A
    8.  Build initial kernel  K0 = X A X^T
    9.  Bregman kernel optimisation
    10. KNN classification  (precomputed kernel)

All output is logged to output1.txt / output2.txt depending on
the dataset choice (1=Olivetti, 2=Brodatz).
"""

import os
import sys
import time
import glob
import warnings
import numpy as np
import cv2

# Suppress harmless logm numerical warnings (err ~1e-13)
warnings.filterwarnings("ignore", message="logm result may be inaccurate")

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)

# ---- make sure we can import sibling modules ----
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from features import extract_features
from covariance import compute_covariance
from log_mapping import log_map, vectorize, geodesic_distance_matrix
from mahalanobis import learn_mahalanobis
from kernel_learning import initial_kernel, gaussian_kernel, log_euclidean_kernel
from bregman import bregman_kernel_learning


# ======================================================================
#  Logging helper: tee output to both console and file
# ======================================================================

class TeeWriter:
    """Write to both stdout and a log file simultaneously."""
    def __init__(self, log_file, original_stdout):
        self.log_file = log_file
        self.original_stdout = original_stdout

    def write(self, message):
        self.original_stdout.write(message)
        self.log_file.write(message)

    def flush(self):
        self.original_stdout.flush()
        self.log_file.flush()




# ======================================================================
#  Helper: convert kernel matrix → KNN distance matrix
# ======================================================================

def kernel_to_distance(K: np.ndarray) -> np.ndarray:
    """
    Convert a kernel (Gram) matrix to a distance matrix.

    d(i,j)^2 = K_ii + K_jj - 2 K_ij

    Negative values are clipped to 0 before taking the square root.
    """
    diag = np.diag(K)
    D2 = diag[:, None] + diag[None, :] - 2 * K
    D2 = np.clip(D2, 0, None)
    return np.sqrt(D2)


# ======================================================================
#  Real dataset loaders
# ======================================================================

def load_olivetti_faces(n_classes: int = 10, samples_per_class: int = 10):
    """
    Load the Olivetti Faces dataset from sklearn (auto-downloaded).

    40 people × 10 grayscale images each, 64×64 pixels.
    We select the first `n_classes` people for a manageable run.

    Returns
    -------
    images : list[np.ndarray]   64×64 grayscale images (float [0,1])
    labels : np.ndarray         Integer class labels
    """
    from sklearn.datasets import fetch_olivetti_faces

    print("    Downloading Olivetti Faces dataset (if not cached) ...")
    data = fetch_olivetti_faces(shuffle=False)

    all_images = data.images     # (400, 64, 64) float64 [0,1]
    all_labels = data.target     # (400,)

    images = []
    labels = []
    for cls in range(n_classes):
        idx = np.where(all_labels == cls)[0][:samples_per_class]
        for i in idx:
            images.append(all_images[i])
            labels.append(cls)

    return images, np.array(labels)


def load_brodatz(data_dir: str, n_classes: int = 5, samples_per_class: int = 10,
                 patch_size: int = 64):
    """
    Load Brodatz textures from a local folder.

    Supports TWO layouts automatically:

    Layout A (USC-SIPI download — flat folder):
        data/textures/1.1.01.tiff   ← class 01 (Grass)
        data/textures/1.1.02.tiff   ← class 02 (Bark)
        ...
        data/textures/1.1.13.tiff   ← class 13 (Plastic bubbles)
        Files named X.X.CC.ext where CC = class number (01-13).

    Layout B (class subfolders):
        data/class1/*.png
        data/class2/*.png
        ...

    Each large image (512×512) is split into non-overlapping 64×64 patches.

    Download: https://sipi.usc.edu/database/database.php?volume=textures
    → Download textures.tar.gz or textures.zip and extract into data/

    Returns
    -------
    images : list[np.ndarray]
    labels : np.ndarray
    """
    import glob

    images = []
    labels = []

    # --- Try Layout A: flat folder with files like 1.1.XX.ext ---
    # Look for image files directly in data_dir (or one level down)
    img_extensions = ["*.tiff", "*.tif", "*.png", "*.jpg", "*.jpeg", "*.bmp", "*.gif"]
    flat_files = []
    for ext in img_extensions:
        flat_files.extend(glob.glob(os.path.join(data_dir, ext)))
        flat_files.extend(glob.glob(os.path.join(data_dir, "**", ext), recursive=True))
    flat_files = sorted(set(flat_files))

    if flat_files:
        # Group files by class ID  (last 2-digit number before extension)
        # e.g. "1.1.05.tiff" → class "05"
        import re
        class_map = {}   # class_id → list of file paths
        for fpath in flat_files:
            basename = os.path.splitext(os.path.basename(fpath))[0]
            # Extract last number group (e.g. "1.1.05" → "05", "D5" → "5")
            nums = re.findall(r'\d+', basename)
            if nums:
                class_id = nums[-1]   # last number = texture class
                class_map.setdefault(class_id, []).append(fpath)

        # Sort classes and take first n_classes
        sorted_classes = sorted(class_map.keys())[:n_classes]

        print(f"    Found {len(flat_files)} image files, {len(class_map)} classes")
        print(f"    Using classes: {sorted_classes}")

        for cls_idx, cls_id in enumerate(sorted_classes):
            patches = []
            for fpath in sorted(class_map[cls_id]):
                img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                H, W = img.shape
                for y in range(0, H - patch_size + 1, patch_size):
                    for x in range(0, W - patch_size + 1, patch_size):
                        patch = img[y:y+patch_size, x:x+patch_size].astype(np.float64) / 255.0
                        patches.append(patch)
            for p in patches[:samples_per_class]:
                images.append(p)
                labels.append(cls_idx)

        return images, np.array(labels) if labels else np.array([], dtype=int)

    # --- Fallback: Layout B — class subfolders ---
    class_dirs = sorted(glob.glob(os.path.join(data_dir, "*")))
    class_dirs = [d for d in class_dirs if os.path.isdir(d)][:n_classes]

    for cls_idx, cls_dir in enumerate(class_dirs):
        files = sorted(glob.glob(os.path.join(cls_dir, "*")))
        patches = []
        for fpath in files:
            img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            H, W = img.shape
            for y in range(0, H - patch_size + 1, patch_size):
                for x in range(0, W - patch_size + 1, patch_size):
                    patch = img[y:y+patch_size, x:x+patch_size].astype(np.float64) / 255.0
                    patches.append(patch)
        for p in patches[:samples_per_class]:
            images.append(p)
            labels.append(cls_idx)

    return images, np.array(labels) if labels else np.array([], dtype=int)


# ======================================================================
#  Main pipeline
# ======================================================================

def main():
    # ------------------------------------------------------------------
    # 0. Choose dataset FIRST (before redirecting output)
    # ------------------------------------------------------------------
    print("=" * 70)
    print(" Kernel Learning on Riemannian Manifold  —  Full Pipeline")
    print("=" * 70)

    print("\n  Select a dataset:")
    print("    1 — Olivetti Faces   (auto-download, 40 people × 10 images, 64×64)")
    print("    2 — Brodatz Textures (local folder required, see instructions.txt)")
    print()

    choice = input("  Enter choice [1/2] (default=1): ").strip()
    if choice == "2":
        DATA_SOURCE = "brodatz"
        output_file = "output2.txt"
    else:
        DATA_SOURCE = "olivetti"
        output_file = "output1.txt"

    # --- Gather additional inputs BEFORE redirecting ---
    if DATA_SOURCE == "olivetti":
        n_classes = 10
        samples_per_class = 10
        BRODATZ_DIR = None
    elif DATA_SOURCE == "brodatz":
        BRODATZ_DIR = input("  Enter path to Brodatz data folder [data/textures]: ").strip() or "data/textures"
        if not glob.glob(os.path.join(BRODATZ_DIR, "*.tiff")) and \
           os.path.isdir(os.path.join(BRODATZ_DIR, "textures")):
            BRODATZ_DIR = os.path.join(BRODATZ_DIR, "textures")
            print(f"    (Auto-detected subfolder: {BRODATZ_DIR})")
        n_classes = int(input("  Number of classes [8]: ").strip() or "8")
        samples_per_class = int(input("  Samples per class [50]: ").strip() or "50")

    # ------------------------------------------------------------------
    # Set up logging: tee all output to the chosen file
    # ------------------------------------------------------------------
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path = os.path.join(project_root, output_file)
    log_fh = open(output_path, "w", encoding="utf-8")
    original_stdout = sys.stdout
    sys.stdout = TeeWriter(log_fh, original_stdout)

    # Record pipeline start time
    pipeline_start = time.time()
    step_times = {}   # step_name → elapsed seconds

    print("=" * 70)
    print(" Kernel Learning on Riemannian Manifold  —  Full Pipeline")
    print("=" * 70)
    print(f"\n  Dataset choice   : {choice} ({DATA_SOURCE})")
    print(f"  Output log file  : {output_file}")
    print(f"  Timestamp        : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  NumPy version    : {np.__version__}")

    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    print(f"\n[1] Loading dataset: {DATA_SOURCE} ...")
    t0 = time.time()

    if DATA_SOURCE == "olivetti":
        images, labels = load_olivetti_faces(n_classes, samples_per_class)
    elif DATA_SOURCE == "brodatz":
        images, labels = load_brodatz(BRODATZ_DIR, n_classes, samples_per_class)
        if len(images) == 0:
            print("\n    ⚠  No images found in Brodatz folder!")
            print(f"       Checked: {os.path.abspath(BRODATZ_DIR)}")
            print("       Expected layout: data/<class1>/*.png, data/<class2>/*.png, ...")
            print("       Download from: https://sipi.usc.edu/database/database.php?volume=textures")
            print("\n    → Falling back to Olivetti Faces dataset ...\n")
            DATA_SOURCE = "olivetti"
            n_classes = 10
            samples_per_class = 10
            images, labels = load_olivetti_faces(n_classes, samples_per_class)

    step_times["1_load_dataset"] = time.time() - t0

    N = len(images)
    n_classes_actual = len(set(labels.tolist()))
    unique_classes = np.unique(labels)

    print(f"    Dataset       : {DATA_SOURCE}")
    print(f"    Total samples : {N}  ({n_classes_actual} classes)")
    print(f"    Image size    : {images[0].shape}")
    print(f"    Time: {step_times['1_load_dataset']:.2f}s")

    # [GRAPH DATA] Samples per class
    print("\n  --- [GRAPH DATA] Samples per class ---")
    for cls in unique_classes:
        count = np.sum(labels == cls)
        print(f"    CLASS_{cls}_SAMPLES = {count}")

    # [GRAPH DATA] Image intensity statistics per class
    print("\n  --- [GRAPH DATA] Image intensity stats per class ---")
    for cls in unique_classes:
        cls_imgs = [images[i] for i in range(N) if labels[i] == cls]
        all_pixels = np.concatenate([img.ravel() for img in cls_imgs])
        print(f"    CLASS_{cls}_MEAN_INTENSITY = {np.mean(all_pixels):.6f}")
        print(f"    CLASS_{cls}_STD_INTENSITY  = {np.std(all_pixels):.6f}")
        print(f"    CLASS_{cls}_MIN_INTENSITY  = {np.min(all_pixels):.6f}")
        print(f"    CLASS_{cls}_MAX_INTENSITY  = {np.max(all_pixels):.6f}")

    # ------------------------------------------------------------------
    # 2. Extract features per image
    # ------------------------------------------------------------------
    print("\n[2] Extracting features (x, y, I, Ix, Iy, Ixx, Iyy) ...")
    t0 = time.time()
    feature_matrices = [extract_features(img) for img in images]
    step_times["2_extract_features"] = time.time() - t0

    d = feature_matrices[0].shape[1]
    n_pixels = feature_matrices[0].shape[0]
    print(f"    Feature dim d = {d}  |  pixels/image = {n_pixels}")
    print(f"    Time: {step_times['2_extract_features']:.2f}s")

    # [GRAPH DATA] Feature statistics (mean, std per feature channel)
    print("\n  --- [GRAPH DATA] Feature channel statistics (averaged over all images) ---")
    channel_names = ["x", "y", "I", "Ix", "Iy", "Ixx", "Iyy"]
    all_feats_stacked = np.vstack(feature_matrices)  # (N*pixels, d)
    for ch_idx, ch_name in enumerate(channel_names):
        ch_mean = np.mean(all_feats_stacked[:, ch_idx])
        ch_std  = np.std(all_feats_stacked[:, ch_idx])
        ch_min  = np.min(all_feats_stacked[:, ch_idx])
        ch_max  = np.max(all_feats_stacked[:, ch_idx])
        print(f"    FEATURE_{ch_name}_MEAN = {ch_mean:.6f}")
        print(f"    FEATURE_{ch_name}_STD  = {ch_std:.6f}")
        print(f"    FEATURE_{ch_name}_MIN  = {ch_min:.6f}")
        print(f"    FEATURE_{ch_name}_MAX  = {ch_max:.6f}")
    del all_feats_stacked  # free memory

    # ------------------------------------------------------------------
    # 3. Compute covariance matrices (SPD)
    # ------------------------------------------------------------------
    print("\n[3] Computing covariance matrices (SPD) ...")
    t0 = time.time()
    covariances = [compute_covariance(F) for F in feature_matrices]
    step_times["3_compute_covariance"] = time.time() - t0

    print(f"    Each covariance shape: {covariances[0].shape}")
    print(f"    Time: {step_times['3_compute_covariance']:.2f}s")

    # [GRAPH DATA] Covariance matrix eigenvalue stats per sample
    print("\n  --- [GRAPH DATA] Covariance eigenvalue statistics (all samples) ---")
    cov_eigvals_all = []
    for idx, C in enumerate(covariances):
        eigs = np.linalg.eigvalsh(C)
        cov_eigvals_all.append(eigs)
    cov_eigvals_all = np.array(cov_eigvals_all)  # (N, d)
    print(f"    COV_EIGVAL_GLOBAL_MIN = {cov_eigvals_all.min():.6e}")
    print(f"    COV_EIGVAL_GLOBAL_MAX = {cov_eigvals_all.max():.6e}")
    print(f"    COV_EIGVAL_GLOBAL_MEAN = {cov_eigvals_all.mean():.6e}")
    for eigdim in range(d):
        print(f"    COV_EIGVAL_DIM{eigdim}_MEAN = {cov_eigvals_all[:, eigdim].mean():.6e}")
        print(f"    COV_EIGVAL_DIM{eigdim}_STD  = {cov_eigvals_all[:, eigdim].std():.6e}")

    # Condition numbers
    cond_numbers = cov_eigvals_all[:, -1] / np.clip(cov_eigvals_all[:, 0], 1e-15, None)
    print(f"    COV_CONDITION_NUM_MEAN = {np.mean(cond_numbers):.4f}")
    print(f"    COV_CONDITION_NUM_MAX  = {np.max(cond_numbers):.4f}")
    print(f"    COV_CONDITION_NUM_MIN  = {np.min(cond_numbers):.4f}")

    # [GRAPH DATA] Covariance matrix trace and determinant
    print("\n  --- [GRAPH DATA] Covariance trace and log-det ---")
    for idx, C in enumerate(covariances):
        tr = np.trace(C)
        sign, logdet = np.linalg.slogdet(C)
        print(f"    SAMPLE_{idx}_COV_TRACE = {tr:.6f}")
        print(f"    SAMPLE_{idx}_COV_LOGDET = {logdet:.6f}")

    # ------------------------------------------------------------------
    # 4. Log-Euclidean mapping
    # ------------------------------------------------------------------
    print("\n[4] Log-Euclidean mapping (matrix logarithm) ...")
    t0 = time.time()
    log_covs = [log_map(C) for C in covariances]
    step_times["4_log_mapping"] = time.time() - t0

    print(f"    log(C) shape: {log_covs[0].shape}")
    print(f"    Time: {step_times['4_log_mapping']:.2f}s")

    # [GRAPH DATA] Log-mapped matrix statistics
    print("\n  --- [GRAPH DATA] Log-mapped matrix statistics ---")
    log_frob_norms = [np.linalg.norm(S, 'fro') for S in log_covs]
    print(f"    LOG_FROB_NORM_MEAN = {np.mean(log_frob_norms):.6f}")
    print(f"    LOG_FROB_NORM_STD  = {np.std(log_frob_norms):.6f}")
    print(f"    LOG_FROB_NORM_MIN  = {np.min(log_frob_norms):.6f}")
    print(f"    LOG_FROB_NORM_MAX  = {np.max(log_frob_norms):.6f}")
    for idx in range(N):
        print(f"    SAMPLE_{idx}_LOG_FROB_NORM = {log_frob_norms[idx]:.6f}")

    # ------------------------------------------------------------------
    # 5. Vectorise and normalise
    # ------------------------------------------------------------------
    print("\n[5] Vectorising log-mapped matrices ...")
    t0 = time.time()
    vectors = np.array([vectorize(S) for S in log_covs])   # (N, d*d)

    # Normalise each feature dimension (zero mean, unit variance)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    vectors = scaler.fit_transform(vectors)
    step_times["5_vectorise"] = time.time() - t0

    print(f"    Descriptor matrix X shape: {vectors.shape}")
    print(f"    (normalised: mean≈0, std≈1 per dimension)")
    print(f"    Time: {step_times['5_vectorise']:.2f}s")

    # [GRAPH DATA] Vectorised descriptor statistics
    print("\n  --- [GRAPH DATA] Vectorised descriptor statistics ---")
    print(f"    VECTOR_GLOBAL_MEAN = {vectors.mean():.6e}")
    print(f"    VECTOR_GLOBAL_STD  = {vectors.std():.6e}")
    print(f"    VECTOR_GLOBAL_MIN  = {vectors.min():.6e}")
    print(f"    VECTOR_GLOBAL_MAX  = {vectors.max():.6e}")

    # Per-sample L2 norms
    sample_norms = np.linalg.norm(vectors, axis=1)
    print(f"    VECTOR_L2NORM_MEAN = {sample_norms.mean():.6f}")
    print(f"    VECTOR_L2NORM_STD  = {sample_norms.std():.6f}")
    for idx in range(N):
        print(f"    SAMPLE_{idx}_VECTOR_L2NORM = {sample_norms[idx]:.6f}")

    # ------------------------------------------------------------------
    # 6. Geodesic distance matrix
    # ------------------------------------------------------------------
    print("\n[6] Computing geodesic distance matrix ...")
    t0 = time.time()
    D_geo = geodesic_distance_matrix(covariances)
    step_times["6_geodesic_distance"] = time.time() - t0

    print(f"    D_geo shape: {D_geo.shape}  |  max dist = {D_geo.max():.4f}")
    print(f"    Time: {step_times['6_geodesic_distance']:.2f}s")

    # [GRAPH DATA] Geodesic distance statistics
    upper_tri = D_geo[np.triu_indices(N, k=1)]
    print("\n  --- [GRAPH DATA] Geodesic distance statistics ---")
    print(f"    GEO_DIST_MEAN   = {np.mean(upper_tri):.6f}")
    print(f"    GEO_DIST_STD    = {np.std(upper_tri):.6f}")
    print(f"    GEO_DIST_MEDIAN = {np.median(upper_tri):.6f}")
    print(f"    GEO_DIST_MIN    = {np.min(upper_tri):.6f}")
    print(f"    GEO_DIST_MAX    = {np.max(upper_tri):.6f}")

    # [GRAPH DATA] Histogram bin counts for geodesic distances
    hist_counts, hist_edges = np.histogram(upper_tri, bins=20)
    print("\n  --- [GRAPH DATA] Geodesic distance histogram (20 bins) ---")
    for b_idx in range(len(hist_counts)):
        print(f"    GEO_HIST_BIN_{b_idx}_RANGE = [{hist_edges[b_idx]:.4f}, {hist_edges[b_idx+1]:.4f})")
        print(f"    GEO_HIST_BIN_{b_idx}_COUNT = {hist_counts[b_idx]}")

    # [GRAPH DATA] Intra-class vs inter-class geodesic distances
    print("\n  --- [GRAPH DATA] Intra-class vs inter-class geodesic distances ---")
    intra_dists = []
    inter_dists = []
    for i in range(N):
        for j in range(i+1, N):
            if labels[i] == labels[j]:
                intra_dists.append(D_geo[i, j])
            else:
                inter_dists.append(D_geo[i, j])
    intra_dists = np.array(intra_dists)
    inter_dists = np.array(inter_dists)
    print(f"    INTRA_CLASS_DIST_MEAN   = {np.mean(intra_dists):.6f}")
    print(f"    INTRA_CLASS_DIST_STD    = {np.std(intra_dists):.6f}")
    print(f"    INTRA_CLASS_DIST_MEDIAN = {np.median(intra_dists):.6f}")
    print(f"    INTER_CLASS_DIST_MEAN   = {np.mean(inter_dists):.6f}")
    print(f"    INTER_CLASS_DIST_STD    = {np.std(inter_dists):.6f}")
    print(f"    INTER_CLASS_DIST_MEDIAN = {np.median(inter_dists):.6f}")
    print(f"    DIST_SEPARABILITY_RATIO = {np.mean(inter_dists) / max(np.mean(intra_dists), 1e-12):.6f}")

    # Per-class intra distances
    for cls in unique_classes:
        cls_idx = np.where(labels == cls)[0]
        cls_dists = []
        for ii in range(len(cls_idx)):
            for jj in range(ii+1, len(cls_idx)):
                cls_dists.append(D_geo[cls_idx[ii], cls_idx[jj]])
        if cls_dists:
            print(f"    CLASS_{cls}_INTRA_DIST_MEAN = {np.mean(cls_dists):.6f}")
            print(f"    CLASS_{cls}_INTRA_DIST_STD  = {np.std(cls_dists):.6f}")

    # ------------------------------------------------------------------
    # 7. Learn Mahalanobis matrix A
    # ------------------------------------------------------------------
    print("\n[7] Learning Mahalanobis matrix A (supervised ITML) ...")
    t0 = time.time()
    A = learn_mahalanobis(vectors, D_geo, labels=labels, n_iters=50, lr=0.1)
    step_times["7_mahalanobis"] = time.time() - t0

    A_eigvals = np.linalg.eigvalsh(A)
    print(f"    A shape: {A.shape}")
    print(f"    Min/Max eigenvalue of A: {A_eigvals.min():.4e} / {A_eigvals.max():.4e}")
    print(f"    Time: {step_times['7_mahalanobis']:.2f}s")

    # [GRAPH DATA] Mahalanobis matrix eigenvalue spectrum
    print("\n  --- [GRAPH DATA] Mahalanobis matrix A eigenvalue spectrum ---")
    for eigidx, eigval in enumerate(A_eigvals):
        print(f"    A_EIGVAL_{eigidx} = {eigval:.6e}")
    print(f"    A_CONDITION_NUMBER = {A_eigvals.max() / max(A_eigvals.min(), 1e-15):.6f}")
    print(f"    A_TRACE      = {np.trace(A):.6f}")
    print(f"    A_FROB_NORM  = {np.linalg.norm(A, 'fro'):.6f}")
    sign, logdet = np.linalg.slogdet(A)
    print(f"    A_LOG_DET    = {logdet:.6f}")

    # ------------------------------------------------------------------
    # 8. Initial kernel K0 = X A X^T
    # ------------------------------------------------------------------
    print("\n[8] Building initial kernel K0 = X A X^T ...")
    t0 = time.time()
    K0 = initial_kernel(vectors, A)
    step_times["8_initial_kernel"] = time.time() - t0

    print(f"    K0 shape: {K0.shape}")
    print(f"    K0 min/max: {K0.min():.4f} / {K0.max():.4f}")
    print(f"    Time: {step_times['8_initial_kernel']:.2f}s")

    # [GRAPH DATA] K0 eigenvalue spectrum
    K0_eigvals = np.linalg.eigvalsh(K0)
    print("\n  --- [GRAPH DATA] Initial kernel K0 eigenvalue spectrum ---")
    print(f"    K0_EIGVAL_MIN  = {K0_eigvals.min():.6e}")
    print(f"    K0_EIGVAL_MAX  = {K0_eigvals.max():.6e}")
    print(f"    K0_EIGVAL_MEAN = {K0_eigvals.mean():.6e}")
    print(f"    K0_TRACE       = {np.trace(K0):.6f}")
    print(f"    K0_FROB_NORM   = {np.linalg.norm(K0, 'fro'):.6f}")
    # Top 10 eigenvalues
    top_k = min(10, len(K0_eigvals))
    for eigidx in range(top_k):
        print(f"    K0_EIGVAL_TOP{eigidx} = {K0_eigvals[-(eigidx+1)]:.6e}")

    # [GRAPH DATA] K0 diagonal statistics
    K0_diag = np.diag(K0)
    print(f"    K0_DIAG_MEAN = {K0_diag.mean():.6f}")
    print(f"    K0_DIAG_STD  = {K0_diag.std():.6f}")
    print(f"    K0_DIAG_MIN  = {K0_diag.min():.6f}")
    print(f"    K0_DIAG_MAX  = {K0_diag.max():.6f}")

    # ------------------------------------------------------------------
    # 9. Bregman kernel optimisation
    # ------------------------------------------------------------------
    print("\n[9] Bregman kernel optimisation ...")
    t0 = time.time()
    K_final = bregman_kernel_learning(K0, D_geo, max_iter=200, verbose=True)
    step_times["9_bregman"] = time.time() - t0

    print(f"    K_final shape: {K_final.shape}")
    print(f"    K_final min/max: {K_final.min():.4f} / {K_final.max():.4f}")
    print(f"    Time: {step_times['9_bregman']:.2f}s")

    # [GRAPH DATA] K_final eigenvalue spectrum
    Kf_eigvals = np.linalg.eigvalsh(K_final)
    print("\n  --- [GRAPH DATA] Bregman kernel K_final eigenvalue spectrum ---")
    print(f"    KF_EIGVAL_MIN  = {Kf_eigvals.min():.6e}")
    print(f"    KF_EIGVAL_MAX  = {Kf_eigvals.max():.6e}")
    print(f"    KF_EIGVAL_MEAN = {Kf_eigvals.mean():.6e}")
    print(f"    KF_TRACE       = {np.trace(K_final):.6f}")
    print(f"    KF_FROB_NORM   = {np.linalg.norm(K_final, 'fro'):.6f}")
    top_k = min(10, len(Kf_eigvals))
    for eigidx in range(top_k):
        print(f"    KF_EIGVAL_TOP{eigidx} = {Kf_eigvals[-(eigidx+1)]:.6e}")

    # [GRAPH DATA] Kernel alignment between K0 and K_final
    k0_frob = np.linalg.norm(K0, 'fro')
    kf_frob = np.linalg.norm(K_final, 'fro')
    alignment = np.sum(K0 * K_final) / (k0_frob * kf_frob + 1e-12)
    print(f"\n    KERNEL_ALIGNMENT_K0_KF = {alignment:.6f}")

    # [GRAPH DATA] Distance preservation check
    D_kernel = kernel_to_distance(K_final)
    d_kernel_upper = D_kernel[np.triu_indices(N, k=1)]
    d_geo_upper = D_geo[np.triu_indices(N, k=1)]
    dist_corr = np.corrcoef(d_kernel_upper, d_geo_upper)[0, 1]
    dist_mae = np.mean(np.abs(d_kernel_upper - d_geo_upper))
    dist_rmse = np.sqrt(np.mean((d_kernel_upper - d_geo_upper) ** 2))
    print(f"\n  --- [GRAPH DATA] Geodesic distance preservation quality ---")
    print(f"    DIST_PRESERVATION_CORR = {dist_corr:.6f}")
    print(f"    DIST_PRESERVATION_MAE  = {dist_mae:.6f}")
    print(f"    DIST_PRESERVATION_RMSE = {dist_rmse:.6f}")

    # ------------------------------------------------------------------
    # 10. KNN Classification
    # ------------------------------------------------------------------
    print("\n[10] KNN Classification ...")

    # Train / test split  (80% train, 20% test within each class)
    train_idx, test_idx = [], []
    for cls in unique_classes:
        cls_indices = np.where(labels == cls)[0]
        split_point = int(len(cls_indices) * 0.8)
        train_idx.extend(cls_indices[:split_point])
        test_idx.extend(cls_indices[split_point:])

    train_idx = np.array(train_idx)
    test_idx  = np.array(test_idx)

    y_train = labels[train_idx]
    y_test  = labels[test_idx]

    print(f"    Train samples: {len(train_idx)}")
    print(f"    Test  samples: {len(test_idx)}")
    print(f"    Train label distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"    Test  label distribution: {dict(zip(*np.unique(y_test, return_counts=True)))}")

    # ==========================================
    #  10a. Bregman-learned kernel
    # ==========================================
    print(f"\n  ---- Bregman Kernel KNN (k=1) ----")
    D_final = kernel_to_distance(K_final)
    D_train = D_final[np.ix_(train_idx, train_idx)]
    D_test  = D_final[np.ix_(test_idx, train_idx)]

    knn = KNeighborsClassifier(n_neighbors=1, metric='precomputed')
    knn.fit(D_train, y_train)
    y_pred = knn.predict(D_test)

    acc_bregman = accuracy_score(y_test, y_pred)
    print(f"  Accuracy : {acc_bregman * 100:.2f}%")
    print(classification_report(y_test, y_pred, zero_division=0))

    # [GRAPH DATA] Per-class accuracy for Bregman
    print("  --- [GRAPH DATA] Bregman per-class metrics ---")
    prec_b, rec_b, f1_b, sup_b = precision_recall_fscore_support(y_test, y_pred, zero_division=0)
    for cls_idx_val, cls_val in enumerate(np.unique(y_test)):
        cls_mask = (y_test == cls_val)
        cls_acc = accuracy_score(y_test[cls_mask], y_pred[cls_mask])
        print(f"    BREGMAN_CLASS_{cls_val}_ACCURACY  = {cls_acc * 100:.2f}")
        print(f"    BREGMAN_CLASS_{cls_val}_PRECISION = {prec_b[cls_idx_val]:.4f}")
        print(f"    BREGMAN_CLASS_{cls_val}_RECALL    = {rec_b[cls_idx_val]:.4f}")
        print(f"    BREGMAN_CLASS_{cls_val}_F1        = {f1_b[cls_idx_val]:.4f}")
        print(f"    BREGMAN_CLASS_{cls_val}_SUPPORT   = {sup_b[cls_idx_val]}")

    # [GRAPH DATA] Confusion matrix for Bregman
    cm_bregman = confusion_matrix(y_test, y_pred)
    print("\n  --- [GRAPH DATA] Bregman confusion matrix ---")
    print(f"    BREGMAN_CONFUSION_MATRIX =")
    for row in cm_bregman:
        print(f"      {row.tolist()}")

    # ==========================================
    #  10b. Gaussian kernel baseline
    # ==========================================
    raw_cov_vectors = np.array([C.flatten() for C in covariances])   # (N, d*d)
    raw_dists = np.sqrt(np.sum((raw_cov_vectors[:, None] - raw_cov_vectors[None, :]) ** 2, axis=-1))
    sigma_gauss = max(np.median(raw_dists[raw_dists > 0]), 1e-6)
    print(f"\n  Gaussian baseline: RBF on RAW covariance vectors (sigma={sigma_gauss:.4f})")
    K_gauss = gaussian_kernel(raw_cov_vectors, sigma=sigma_gauss)
    D_gauss = kernel_to_distance(K_gauss)
    D_train_g = D_gauss[np.ix_(train_idx, train_idx)]
    D_test_g  = D_gauss[np.ix_(test_idx, train_idx)]

    knn_g = KNeighborsClassifier(n_neighbors=1, metric='precomputed')
    knn_g.fit(D_train_g, y_train)
    y_pred_g = knn_g.predict(D_test_g)
    acc_gauss = accuracy_score(y_test, y_pred_g)

    print(f"\n  ---- Gaussian Kernel Baseline (k=1) ----")
    print(f"  Accuracy : {acc_gauss * 100:.2f}%")
    print(classification_report(y_test, y_pred_g, zero_division=0))

    # [GRAPH DATA] Per-class accuracy for Gaussian
    print("  --- [GRAPH DATA] Gaussian per-class metrics ---")
    prec_g, rec_g, f1_g, sup_g = precision_recall_fscore_support(y_test, y_pred_g, zero_division=0)
    for cls_idx_val, cls_val in enumerate(np.unique(y_test)):
        cls_mask = (y_test == cls_val)
        cls_acc = accuracy_score(y_test[cls_mask], y_pred_g[cls_mask])
        print(f"    GAUSSIAN_CLASS_{cls_val}_ACCURACY  = {cls_acc * 100:.2f}")
        print(f"    GAUSSIAN_CLASS_{cls_val}_PRECISION = {prec_g[cls_idx_val]:.4f}")
        print(f"    GAUSSIAN_CLASS_{cls_val}_RECALL    = {rec_g[cls_idx_val]:.4f}")
        print(f"    GAUSSIAN_CLASS_{cls_val}_F1        = {f1_g[cls_idx_val]:.4f}")
        print(f"    GAUSSIAN_CLASS_{cls_val}_SUPPORT   = {sup_g[cls_idx_val]}")

    # [GRAPH DATA] Confusion matrix for Gaussian
    cm_gauss = confusion_matrix(y_test, y_pred_g)
    print("\n  --- [GRAPH DATA] Gaussian confusion matrix ---")
    print(f"    GAUSSIAN_CONFUSION_MATRIX =")
    for row in cm_gauss:
        print(f"      {row.tolist()}")

    # ==========================================
    #  10c. Log-Euclidean kernel baseline
    # ==========================================
    sigma_le = max(np.median(D_geo[D_geo > 0]), 0.1)
    print(f"\n  Log-Euclidean baseline: RBF on log-mapped vectors (sigma={sigma_le:.4f})")
    K_le = log_euclidean_kernel(covariances, sigma=sigma_le)
    D_le = kernel_to_distance(K_le)
    D_train_le = D_le[np.ix_(train_idx, train_idx)]
    D_test_le  = D_le[np.ix_(test_idx, train_idx)]

    knn_le = KNeighborsClassifier(n_neighbors=1, metric='precomputed')
    knn_le.fit(D_train_le, y_train)
    y_pred_le = knn_le.predict(D_test_le)
    acc_le = accuracy_score(y_test, y_pred_le)

    print(f"\n  ---- Log-Euclidean Kernel Baseline (k=1) ----")
    print(f"  Accuracy : {acc_le * 100:.2f}%")
    print(classification_report(y_test, y_pred_le, zero_division=0))

    # [GRAPH DATA] Per-class accuracy for Log-Euclidean
    print("  --- [GRAPH DATA] Log-Euclidean per-class metrics ---")
    prec_le, rec_le, f1_le, sup_le = precision_recall_fscore_support(y_test, y_pred_le, zero_division=0)
    for cls_idx_val, cls_val in enumerate(np.unique(y_test)):
        cls_mask = (y_test == cls_val)
        cls_acc = accuracy_score(y_test[cls_mask], y_pred_le[cls_mask])
        print(f"    LOGEUC_CLASS_{cls_val}_ACCURACY  = {cls_acc * 100:.2f}")
        print(f"    LOGEUC_CLASS_{cls_val}_PRECISION = {prec_le[cls_idx_val]:.4f}")
        print(f"    LOGEUC_CLASS_{cls_val}_RECALL    = {rec_le[cls_idx_val]:.4f}")
        print(f"    LOGEUC_CLASS_{cls_val}_F1        = {f1_le[cls_idx_val]:.4f}")
        print(f"    LOGEUC_CLASS_{cls_val}_SUPPORT   = {sup_le[cls_idx_val]}")

    # [GRAPH DATA] Confusion matrix for Log-Euclidean
    cm_le = confusion_matrix(y_test, y_pred_le)
    print("\n  --- [GRAPH DATA] Log-Euclidean confusion matrix ---")
    print(f"    LOGEUC_CONFUSION_MATRIX =")
    for row in cm_le:
        print(f"      {row.tolist()}")

    # ==========================================
    #  10d. K-sweep: accuracy vs K for all kernels
    # ==========================================
    print("\n  --- [GRAPH DATA] K-sweep: accuracy vs number of neighbors ---")
    max_k = min(15, len(train_idx) - 1)
    for k_val in range(1, max_k + 1, 2):  # k=1,3,5,...
        # Bregman
        knn_b = KNeighborsClassifier(n_neighbors=k_val, metric='precomputed')
        knn_b.fit(D_train, y_train)
        acc_b_k = accuracy_score(y_test, knn_b.predict(D_test))

        # Gaussian
        knn_g_k = KNeighborsClassifier(n_neighbors=k_val, metric='precomputed')
        knn_g_k.fit(D_train_g, y_train)
        acc_g_k = accuracy_score(y_test, knn_g_k.predict(D_test_g))

        # Log-Euclidean
        knn_le_k = KNeighborsClassifier(n_neighbors=k_val, metric='precomputed')
        knn_le_k.fit(D_train_le, y_train)
        acc_le_k = accuracy_score(y_test, knn_le_k.predict(D_test_le))

        print(f"    K={k_val:2d} | BREGMAN_ACC={acc_b_k*100:.2f}% | GAUSSIAN_ACC={acc_g_k*100:.2f}% | LOGEUC_ACC={acc_le_k*100:.2f}%")

    # ==========================================
    #  Kernel matrix statistics comparison
    # ==========================================
    print("\n  --- [GRAPH DATA] Kernel matrix statistics comparison ---")
    for k_name, K_mat in [("BREGMAN", K_final), ("GAUSSIAN", K_gauss), ("LOGEUC", K_le)]:
        k_eigs = np.linalg.eigvalsh(K_mat)
        print(f"    {k_name}_KERNEL_TRACE     = {np.trace(K_mat):.6f}")
        print(f"    {k_name}_KERNEL_FROB_NORM = {np.linalg.norm(K_mat, 'fro'):.6f}")
        print(f"    {k_name}_KERNEL_MEAN      = {K_mat.mean():.6f}")
        print(f"    {k_name}_KERNEL_STD       = {K_mat.std():.6f}")
        print(f"    {k_name}_KERNEL_MIN       = {K_mat.min():.6f}")
        print(f"    {k_name}_KERNEL_MAX       = {K_mat.max():.6f}")
        print(f"    {k_name}_KERNEL_EIGMIN    = {k_eigs.min():.6e}")
        print(f"    {k_name}_KERNEL_EIGMAX    = {k_eigs.max():.6e}")
        print(f"    {k_name}_KERNEL_RANK      = {np.sum(k_eigs > 1e-10)}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    pipeline_total = time.time() - pipeline_start
    step_times["total_pipeline"] = pipeline_total

    print("\n" + "=" * 70)
    print(" SUMMARY")
    print("=" * 70)
    print(f"  Dataset                   : {DATA_SOURCE}")
    print(f"  Total samples             : {N}  ({n_classes_actual} classes)")
    print(f"  Final kernel matrix shape : {K_final.shape}")
    print(f"  Bregman Kernel Accuracy   : {acc_bregman * 100:.2f}%")
    print(f"  Gaussian Kernel Accuracy  : {acc_gauss * 100:.2f}%")
    print(f"  Log-Euclidean Accuracy    : {acc_le * 100:.2f}%")
    print(f"  Total pipeline time       : {pipeline_total:.2f}s")

    # [GRAPH DATA] Accuracy bar chart data
    print("\n  --- [GRAPH DATA] Final accuracy comparison ---")
    print(f"    FINAL_BREGMAN_ACCURACY  = {acc_bregman * 100:.2f}")
    print(f"    FINAL_GAUSSIAN_ACCURACY = {acc_gauss * 100:.2f}")
    print(f"    FINAL_LOGEUC_ACCURACY   = {acc_le * 100:.2f}")

    # [GRAPH DATA] Timing breakdown
    print("\n  --- [GRAPH DATA] Step timing breakdown (seconds) ---")
    for step_name, elapsed in step_times.items():
        print(f"    TIME_{step_name.upper()} = {elapsed:.4f}")

    # [GRAPH DATA] Sigma parameters used
    print("\n  --- [GRAPH DATA] Kernel parameters ---")
    print(f"    GAUSSIAN_SIGMA  = {sigma_gauss:.6f}")
    print(f"    LOGEUC_SIGMA    = {sigma_le:.6f}")

    print("=" * 70)
    print(f"   All output saved to: {output_path}")
    print("=" * 70)

    # Restore stdout and close log file
    sys.stdout = original_stdout
    log_fh.close()

    print(f"\n  Log saved to: {output_path}")


if __name__ == "__main__":
    main()
