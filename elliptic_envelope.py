"""
Elliptic Envelope Outlier Detection for Diabetic Patient Data
CMPT 459 Course Project

Uses PCA (50 components) + Elliptic Envelope (custom),
visualizes outliers with PCA, and summarizes results.
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA


# ============================================================
# Custom Elliptic Envelope (uses covariance of reduced data)
# ============================================================
class EllipticEnvelopeCustom:
    """
    Simple Elliptic Envelope based on Mahalanobis distance.

    NOTE: Must be used on PCA-reduced data (e.g., 50 dims),
    otherwise the covariance matrix becomes too large.
    """

    def __init__(self, contamination=0.05):
        assert 0 < contamination < 0.5
        self.contamination = contamination
        self.location_ = None
        self.cov_inv_ = None
        self.threshold_ = None

    def fit(self, X):
        # Estimate mean
        self.location_ = X.mean(axis=0)

        # Compute covariance on PCA-reduced data (safe!)
        cov = np.cov(X, rowvar=False)
        cov += 1e-6 * np.eye(cov.shape[0])
        self.cov_inv_ = np.linalg.inv(cov)

        # Mahalanobis distances
        d = self._mahalanobis(X)

        # threshold
        self.threshold_ = np.quantile(d, 1 - self.contamination)
        return self

    def _mahalanobis(self, X):
        diff = X - self.location_
        m = np.einsum("ij,jk,ik->i", diff, self.cov_inv_, diff)
        m = np.maximum(m, 0.0)
        return np.sqrt(m)

    def decision_function(self, X):
        d = self._mahalanobis(X)
        return self.threshold_ - d

    def predict(self, X):
        scores = self.decision_function(X)
        return np.where(scores >= 0, 1, -1)


# ============================================================
# Data loader (same as your pipeline)
# ============================================================
def load_and_preprocess(path):
    print("Loading data...")
    df = pd.read_csv(path)
    print(f"Original shape: {df.shape}")

    df = df.replace("?", np.nan)

    # Drop >40% missing
    threshold = 0.4 * len(df)
    df = df.dropna(thresh=threshold, axis=1)

    # Fill categorical NAs
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna("Unknown")

    # Encode target
    if "readmitted" in df.columns:
        df["readmitted"] = df["readmitted"].map({"NO": 0, ">30": 1, "<30": 2})
        y = df["readmitted"].copy()
        df = df.drop(columns=["readmitted"])
    else:
        y = None

    # Remove IDs
    for col in ["encounter_id", "patient_nbr"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Encode categorical
    cat_cols = df.select_dtypes(include="object").columns
    le = LabelEncoder()

    for col in cat_cols:
        if df[col].nunique() < 10:
            df[col] = le.fit_transform(df[col].astype(str))
        else:
            df = pd.get_dummies(df, columns=[col], drop_first=True)

    # Scale numeric
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    X = df.values
    print(f"Final feature matrix shape: {X.shape}")
    print("Preprocessing complete!\n")
    return X, y


# ============================================================
# Visualization
# ============================================================
def visualize_outliers(X_orig, preds, scores, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    print("Running PCA for visualization (2D)...")
    pca_vis = PCA(n_components=2, random_state=42)
    X_vis = pca_vis.fit_transform(X_orig)

    inlier_mask = preds == 1
    outlier_mask = preds == -1

    # Scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(X_vis[inlier_mask, 0], X_vis[inlier_mask, 1],
                c='blue', alpha=0.3, s=20, label='Inliers')
    plt.scatter(X_vis[outlier_mask, 0], X_vis[outlier_mask, 1],
                c='red', alpha=0.8, s=40, marker='x', label='Outliers')
    plt.title("Elliptic Envelope Outliers (PCA 2D)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(alpha=0.3)
    path1 = os.path.join(output_dir, "elliptic_outliers_pca.png")
    plt.tight_layout()
    plt.savefig(path1, dpi=300)
    plt.close()
    print(f"Saved PCA outlier plot to {path1}")

    # Histogram
    plt.figure(figsize=(8, 5))
    plt.hist(scores, bins=50, edgecolor="black", alpha=0.7)
    plt.axvline(0.0, color='red', linestyle='--', label='Decision boundary (0)')
    plt.title("Decision Function Scores")
    plt.xlabel("Score (threshold - distance)")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(alpha=0.3)
    path2 = os.path.join(output_dir, "elliptic_scores_hist.png")
    plt.tight_layout()
    plt.savefig(path2, dpi=300)
    plt.close()
    print(f"Saved score histogram to {path2}")


# ============================================================
# Argument parsing
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Elliptic Envelope Outlier Detection")
    parser.add_argument("--data", type=str, default="data/diabetic_data.csv")
    parser.add_argument("--contamination", type=float, default=0.05)
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--pca-components", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default="elliptic_outlier_plots")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


# ============================================================
# Main
# ============================================================
def main():
    args = parse_args()
    np.random.seed(args.random_state)

    print("=" * 70)
    print("ELLIPTIC ENVELOPE OUTLIER DETECTION")
    print("=" * 70)
    print(f"Dataset:         {args.data}")
    print(f"Contamination:   {args.contamination}")
    print(f"Sample size:     {args.sample_size}")
    print(f"PCA components:  {args.pca_components}")
    print(f"Output dir:      {args.output_dir}")
    print("=" * 70)

    # Load dataset
    X, y = load_and_preprocess(args.data)

    # Sampling
    if X.shape[0] > args.sample_size:
        print(f"Sampling {args.sample_size} points...")
        idx = np.random.choice(X.shape[0], args.sample_size, replace=False)
        X = X[idx]
        if y is not None:
            y = np.array(y)[idx]

    # PCA reduction BEFORE Elliptic Envelope
    print("\nApplying PCA reduction...")
    pca = PCA(n_components=args.pca_components, random_state=42)
    X_pca = pca.fit_transform(X)
    print(f"PCA complete. New shape: {X_pca.shape}")
    print(f"Explained variance: {pca.explained_variance_ratio_.sum()*100:.2f}%")

    # Fit Elliptic Envelope
    print("\nFitting Elliptic Envelope...")
    ee = EllipticEnvelopeCustom(contamination=args.contamination)
    ee.fit(X_pca)
    preds = ee.predict(X_pca)
    scores = ee.decision_function(X_pca)

    n_outliers = int(np.sum(preds == -1))
    print("\nRESULTS")
    print("-" * 70)
    print(f"Samples:          {X.shape[0]}")
    print(f"Outliers:         {n_outliers} ({n_outliers / len(X) * 100:.2f}%)")
    print(f"Score range:      [{scores.min():.4f}, {scores.max():.4f}]")

    # Visualize outliers ON ORIGINAL SPACE (2D PCA)
    visualize_outliers(X, preds, scores, args.output_dir)

    print("\nDone!")
    print(f"Plots saved to: {args.output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
