📘 README.md
Kernel Learning on Riemannian Manifold (Geodesic Distance Preservation)
📌 Goal

Implement the algorithm from the paper:

Kernel Learning Method on Riemannian Manifold with Geodesic Distance Preservation

This method:

Converts data → SPD covariance matrices
Uses log-Euclidean mapping
Learns Mahalanobis distance
Builds an optimal kernel matrix
Preserves geodesic distance
🧠 Core Idea (VERY IMPORTANT)

We want:

Distance in kernel space = Geodesic distance on manifold

Pipeline:

Image → Features → Covariance (SPD)
→ log(S) → vector → Mahalanobis distance
→ Kernel matrix → Bregman optimization
→ Final kernel
📦 Requirements
numpy
scipy
opencv-python
scikit-learn
📁 Folder Structure
project/
│── data/
│── src/
│   ├── features.py
│   ├── covariance.py
│   ├── log_mapping.py
│   ├── mahalanobis.py
│   ├── kernel_learning.py
│   ├── bregman.py
│   └── main.py
🔧 IMPLEMENTATION STEPS
1️⃣ Feature Extraction

Extract features per pixel:

x, y coordinates
intensity (I)
gradients: Ix, Iy
second derivatives: Ixx, Iyy
def extract_features(img):
    # shape: (H, W)
    # output: (H*W, d)

👉 Output: Feature matrix Z ∈ R^(n × d)

2️⃣ Covariance Matrix (SPD)

From paper Eq.(2)(3):

def compute_covariance(features):
    # features: (n, d)
    mean = np.mean(features, axis=0)
    centered = features - mean
    C = centered.T @ centered / len(features)
    return C

👉 Output: SPD matrix (d × d)

3️⃣ Log-Euclidean Mapping

Convert SPD → Euclidean space:

from scipy.linalg import logm

def log_map(C):
    return logm(C)
4️⃣ Vectorization

Flatten matrix → vector

def vectorize(S):
    return S.flatten()
5️⃣ Geodesic Distance (Log-Euclidean)

From paper:

d(S1, S2) = || log(S1) - log(S2) ||
def geodesic_distance(S1, S2):
    return np.linalg.norm(logm(S1) - logm(S2))
6️⃣ Mahalanobis Distance Learning

We want:

d_A(xi, xj)^2 = (xi - xj)^T A (xi - xj)

And match it with geodesic distance.

Optimization Goal:

Minimize LogDet divergence:

D(A, A0) = tr(A A0⁻¹) - log det(A A0⁻¹) - n

👉 Simplified approach:

def learn_mahalanobis(X, D_geo):
    # X: vectors
    # D_geo: geodesic distances matrix

    d = X.shape[1]
    A = np.eye(d)

    # Iterative optimization (simplified)
    for _ in range(100):
        # update rule (approximate)
        pass

    return A

⚠️ NOTE: Paper uses LogDet + Bregman optimization (iterative)

7️⃣ Initial Kernel Matrix

From paper:

K0 = X A X^T
def initial_kernel(X, A):
    return X @ A @ X.T
8️⃣ Bregman Optimization (Kernel Learning)

Goal:

min D(K, K0)
subject to:
Kii + Kjj - 2Kij = d_geodesic(i,j)^2
Iterative Algorithm (IMPORTANT)
def bregman_kernel_learning(K0, D_geo, max_iter=100):
    K = K0.copy()
    n = K.shape[0]

    for _ in range(max_iter):
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue

                # compute constraint error
                dist = K[i,i] + K[j,j] - 2*K[i,j]
                target = D_geo[i,j]**2

                # update rule (simplified)
                delta = (target - dist) / 4

                K[i,i] += delta
                K[j,j] += delta
                K[i,j] -= delta
                K[j,i] -= delta

    return K
9️⃣ Classification (KNN)
from sklearn.neighbors import KNeighborsClassifier

def classify(K_train, K_test, y_train):
    knn = KNeighborsClassifier(n_neighbors=3, metric='precomputed')
    knn.fit(K_train, y_train)
    return knn.predict(K_test)
🚀 MAIN PIPELINE
# 1. Load images
# 2. Extract features
# 3. Compute covariance matrices
# 4. Log map
# 5. Vectorize
# 6. Compute geodesic distance matrix
# 7. Learn Mahalanobis matrix A
# 8. Compute initial kernel K0
# 9. Optimize kernel (Bregman)
# 10. Train + test using KNN
⚠️ Important Notes
Covariance matrices must be SPD
Use log-Euclidean metric (simpler than affine-invariant)
Kernel must remain positive definite
Optimization is iterative and sensitive
🧪 Dataset (From Paper)
Brodatz texture dataset
Image size: 640×640
Split:
Train: 50 samples/class
Test: 50 samples/class
🧩 Tips for Implementation
Add small ε to covariance diagonal → ensure SPD
Normalize features before covariance
Use scipy.linalg.logm carefully (slow)
Vector size = d² (can be large)
🎯 Expected Output
Kernel matrix K
Classification accuracy (KNN)
Compare with:
Gaussian kernel
Log-Euclidean kernel