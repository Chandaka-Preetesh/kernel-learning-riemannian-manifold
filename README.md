# 📘 Kernel Learning on Riemannian Manifold

## 📌 Overview
This project implements a kernel learning framework on **Riemannian manifolds** using **SPD covariance matrices**.  
The objective is to learn a kernel that preserves **geodesic distances**, enabling better representation of non-Euclidean data.

---

## 🧠 Core Idea
Distance in kernel space should match the **geodesic distance on the manifold**.

---

## 🔁 Pipeline
Image → Features → Covariance (SPD) → log(C) → Vector  
→ Mahalanobis Learning → Kernel (K0) → Bregman Optimization  
→ Final Kernel → KNN Classification

---

## ⚙️ Features
- Log-Euclidean mapping of SPD matrices  
- Mahalanobis distance learning  
- Kernel optimization using Bregman projection  
- Geodesic distance preservation  
- KNN classification using learned kernel  

---

## 📦 Requirements
Install dependencies using:
pip install -r requirements.txt

---

## 🚀 Run
python src/main.py

---

## 🎯 Output
- Optimized kernel matrix  
- Classification accuracy  
- Output logs in OUTPUT_TEXT_FILES/  
- Visual results in OUTPUT_IMAGES/  

---

## ⚠️ Notes
- Covariance matrices are regularized to ensure SPD  
- Log-Euclidean metric is used for stability  
- Optimization is iterative and sensitive  

---

## 📄 Reference
Kernel Learning Method on Riemannian Manifold with Geodesic Distance Preservation
