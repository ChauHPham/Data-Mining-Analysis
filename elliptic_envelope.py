"""
Elliptic Envelope Outlier Detection for Diabetic Patient Data
CMPT 459 Course Project

Performs a single outlier detection method (Elliptic Envelope),
visualizes outliers with PCA, and summarizes results.
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA


class EllipticEnvelopeCustom:
    """
    Simple elliptical outlier detector.
    Approximates Elliptic Envelope using classical mean + covariance.
    """

    def __init__(self, contamination: float = 0.05):
        """
        :param contamination: proportion of points expected to be outliers (0 < c < 0.5)
        """
        assert 0 < contamination < 0.5
        self.contamination = contamination
        self.location_ = None      # mean vector
        self.cov_inv_ = None       # inverse covariance
        self.threshold_ = None     # Mahalanobis distance cutoff

    def fit(self, X: np.ndarray):
        """
        Fit elliptical model on data X.

        :param X: array of shape (n_samples, n_features)
        """
        # Estimate location and covariance
        self.location_ = X.mean(axis=0)
        cov = np.cov(X, rowvar=False)
        # Regularize covariance slightly for numerical stability
        cov += 1e-6 * np.eye(cov.shape[0])
        self.cov_inv_ = np.linalg.inv(cov)

        # Compute Mahalanobis distances
        d = self._mahalanobis(X)

        # Set threshold based on contamination proportion
        self.threshold_ = np.quantile(d, 1 - self.contamination)
        return self

    def _mahalanobis(self, X: np.ndarray) -> np.ndarray:
        """
        Compute Mahalanobis distance of each point in X from the fitted ellipse.
        """
        diff = X - self.location_
        m = np.einsum("ij,jk,ik->i", diff, self.cov_inv_, diff)
        m = np.maximum(m, 0.0)
        return np.sqrt(m)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Smaller values => more inlier-like, larger => more outlier-like.
        Returns (threshold - distance): positive = inlier, negative = outlier.
        """
        d = self._mahalanobis(X)
        return self.threshold_ - d

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict inliers (+1) and outliers (-1).
        """
        scores = self.decision_function(X)
        labels = np.where(scores >= 0, 1, -1)
        return labels


# -----------------------------------------------------------
# Data loading / preprocessing (mirrors EDA notebook)
# -----------------------------------------------------------
def load_and_preprocess_data(path: str):
    print("Loading data...")
    df = pd.read_csv(path)
    print(f"Original shape: {df.shape}")

    # Replace '?' placeholders with NaN
    df = df.replace('?', np.nan)

    # Drop columns with >40% missing values
    threshold = 0.4 * len(df)
    df = df.dropna(thresh=threshold, axis=1)

    # Fill remaining missing values for categorical features with 'Unknown'
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna('Unknown')

    # Map target variable 'readmitted' (we won't use it as features)
    if 'readmitted' in df.columns:
        df['readmitted'] = df['readmitted'].map({'NO': 0, '>30': 1, '<30': 2})
        y = df['readmitted'].copy()
        df = df.drop(columns=['readmitted'])
    else:
        y = None

    # Drop obvious ID columns if they exist
    id_cols = ['encounter_id', 'patient_nbr']
    for col in id_cols:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Encode categorical variables
    cat_cols = df.select_dtypes(include='object').columns
    le = LabelEncoder()

    for col in cat_cols:
        if df[col].nunique() < 10:
            df[col] = le.fit_transform(df[col].astype(str))
        else:
            df = pd.get_dummies(df, columns=[col], drop_first=True)

    # Normalize numeric features
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    X = df.values
    feature_names = df.columns.to_list()

    print(f"Final feature matrix shape: {X.shape}")
    print("Preprocessing complete!\n")
    return X, y, feature_names


def visualize_outliers_pca(X, preds, scores, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    print("Applying PCA for 2D visualization...")
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    explained = pca.explained_variance_ratio_.sum()
    print(f"PCA explained variance: {explained:.2%}")

    inlier_mask = preds == 1
    outlier_mask = preds == -1

    # Scatter plot with outliers marked
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[inlier_mask, 0], X_pca[inlier_mask, 1],
                c='blue', alpha=0.3, s=20, label='Inliers')
    plt.scatter(X_pca[outlier_mask, 0], X_pca[outlier_mask, 1],
                c='red', alpha=0.8, s=40, marker='x', label='Outliers')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Elliptic Envelope: Outliers vs Inliers (PCA 2D)')
    plt.legend()
    plt.grid(alpha=0.3)
    out_path = os.path.join(output_dir, 'elliptic_envelope_outliers_pca.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved PCA outlier plot to {out_path}")

    # Histogram of anomaly scores (decision_function)
    plt.figure(figsize=(8, 5))
    plt.hist(scores, bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(0.0, color='red', linestyle='--', label='Decision boundary (0)')
    plt.xlabel('Decision Function Value (threshold - distance)')
    plt.ylabel('Frequency')
    plt.title('Elliptic Envelope Anomaly Score Distribution')
    plt.legend()
    plt.grid(alpha=0.3)
    hist_path = os.path.join(output_dir, 'elliptic_envelope_scores_hist.png')
    plt.tight_layout()
    plt.savefig(hist_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved score histogram to {hist_path}")


def parse_args():
    parser = argparse.ArgumentParser(description='Elliptic Envelope Outlier Detection')
    parser.add_argument('--data', type=str, default='data/diabetic_data.csv',
                        help='Path to the dataset CSV')
    parser.add_argument('--contamination', type=float, default=0.05,
                        help='Expected proportion of outliers (0.0–0.5)')
    parser.add_argument('--sample-size', type=int, default=5000,
                        help='Max number of samples to use (for speed)')
    parser.add_argument('--output-dir', type=str, default='elliptic_outlier_plots',
                        help='Directory for saving plots')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random seed for reproducibility')
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.random_state)

    print("=" * 70)
    print("ELLIPTIC ENVELOPE OUTLIER DETECTION")
    print("=" * 70)
    print(f"Dataset:        {args.data}")
    print(f"Contamination:  {args.contamination:.3f}")
    print(f"Sample size:    {args.sample_size}")
    print(f"Output dir:     {args.output_dir}")
    print("=" * 70)

    # Load & preprocess
    X, y, feature_names = load_and_preprocess_data(args.data)

    # Sample if necessary
    if X.shape[0] > args.sample_size:
        print(f"Sampling {args.sample_size} points for speed...")
        idx = np.random.choice(X.shape[0], args.sample_size, replace=False)
        X = X[idx]
        if y is not None:
            y = np.array(y)[idx]

    # Fit Elliptic Envelope
    ee = EllipticEnvelopeCustom(contamination=args.contamination)
    ee.fit(X)
    preds = ee.predict(X)
    scores = ee.decision_function(X)

    n_outliers = int(np.sum(preds == -1))
    n_samples = X.shape[0]

    print("\nRESULTS")
    print("-" * 70)
    print(f"Total samples used:   {n_samples}")
    print(f"Detected outliers:    {n_outliers}  ({n_outliers / n_samples * 100:.2f}%)")
    print(f"Decision function range: [{scores.min():.4f}, {scores.max():.4f}]")
    print("-" * 70)
    print("Interpretation:")
    print("  • Positive scores ( > 0 )   → inliers (inside ellipse)")
    print("  • Negative scores ( < 0 )   → outliers (outside ellipse)")
    print("  • You should inspect outliers to decide whether they are noise")
    print("    or rare but important patient types.\n")

    # Visualization
    visualize_outliers_pca(X, preds, scores, args.output_dir)

    print("\nDone. Plots saved in:", args.output_dir)
    print("=" * 70)


if __name__ == '__main__':
    main()
