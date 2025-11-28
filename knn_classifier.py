"""
k-NN Classification for Diabetic Readmission
CMPT 459 Course Project

Trains a KNN classifier, evaluates on a test split,
prints metrics, and saves confusion matrix + PCA visualization.
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

from collections import Counter


class KNNClassifier:
    """
    Simple k-Nearest Neighbors classifier using NumPy.
    """

    def __init__(self, n_neighbors: int = 5, weights: str = "uniform", p: int = 2):
        """
        :param n_neighbors: number of neighbors (k)
        :param weights: 'uniform' or 'distance'
        :param p: 1 for Manhattan, 2 for Euclidean
        """
        assert n_neighbors >= 1
        assert weights in ("uniform", "distance")
        assert p in (1, 2)
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.p = p
        self.X_train = None
        self.y_train = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X_train = X
        self.y_train = y
        return self

    def _minkowski_distances(self, X: np.ndarray) -> np.ndarray:
        if self.p == 2:
            X_sq = np.sum(X ** 2, axis=1).reshape(-1, 1)
            Xtr_sq = np.sum(self.X_train ** 2, axis=1)
            dist_sq = X_sq + Xtr_sq - 2 * X @ self.X_train.T
            dist_sq = np.maximum(dist_sq, 0.0)
            return np.sqrt(dist_sq)
        else:
            return np.sum(
                np.abs(X[:, None, :] - self.X_train[None, :, :]),
                axis=2
            )

    def _predict_one(self, dist_row: np.ndarray) -> int:
        nn_idx = np.argpartition(dist_row, self.n_neighbors)[:self.n_neighbors]
        nn_labels = self.y_train[nn_idx]
        nn_dists = dist_row[nn_idx]

        if self.weights == "uniform":
            counts = Counter(nn_labels)
            return counts.most_common(1)[0][0]
        else:
            eps = 1e-8
            w = 1.0 / (nn_dists + eps)
            scores = {}
            for lbl, weight in zip(nn_labels, w):
                scores[lbl] = scores.get(lbl, 0.0) + weight
            return max(scores.items(), key=lambda kv: kv[1])[0]

    def predict(self, X: np.ndarray) -> np.ndarray:
        dists = self._minkowski_distances(X)
        y_pred = np.apply_along_axis(self._predict_one, 1, dists)
        return y_pred

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)
        return float(np.mean(y_pred == y))


# -----------------------------------------------------------
# Data preprocessing helper
# -----------------------------------------------------------
def load_and_preprocess_data(path: str):
    print("Loading data...")
    df = pd.read_csv(path)
    print(f"Original shape: {df.shape}")

    df = df.replace('?', np.nan)

    # Drop columns with >40% missing
    threshold = 0.4 * len(df)
    df = df.dropna(thresh=threshold, axis=1)

    # Fill categorical NAs
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna("Unknown")

    # Encode target
    df["readmitted"] = df["readmitted"].map({'NO': 0, '>30': 1, '<30': 2})

    # Encode categorical
    cat_cols = df.select_dtypes(include='object').columns
    le = LabelEncoder()
    for col in cat_cols:
        if df[col].nunique() < 10:
            df[col] = le.fit_transform(df[col].astype(str))
        else:
            df = pd.get_dummies(df, columns=[col], drop_first=True)

    # Drop IDs
    for col in ["encounter_id", "patient_nbr"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Scale numeric
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    X = df.drop(columns=["readmitted"]).values
    y = df["readmitted"].values
    print("Preprocessing complete! Final shape:", X.shape)
    return X, y


def plot_confusion_matrix(cm, class_names, out_path):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('k-NN Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, int(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix to {out_path}")


def plot_pca_scatter(X, y_true, y_pred, out_path):
    print("Computing 2D PCA for visualization...")
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)

    correct = (y_true == y_pred)
    plt.figure(figsize=(8, 6))
    # Correct predictions
    plt.scatter(X_pca[correct, 0], X_pca[correct, 1],
                c=y_pred[correct], cmap='viridis',
                alpha=0.5, s=20, label='Correct')
    # Misclassified
    plt.scatter(X_pca[~correct, 0], X_pca[~correct, 1],
                c=y_pred[~correct], cmap='viridis',
                alpha=0.9, s=60, marker='x', label='Misclassified', edgecolors='black')

    plt.title('k-NN Predictions in PCA Space\n(colored by predicted class)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved PCA scatter to {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description='k-NN Classification for Diabetic Readmission')
    parser.add_argument('--data', type=str, default='data/diabetic_data.csv',
                        help='Path to dataset CSV')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Test set proportion')
    parser.add_argument('--n-neighbors', type=int, default=5,
                        help='Number of neighbors (k)')
    parser.add_argument('--weights', type=str, default='uniform',
                        choices=['uniform', 'distance'],
                        help='Weight function')
    parser.add_argument('--p', type=int, default=2, choices=[1, 2],
                        help='Minkowski p (1=Manhattan, 2=Euclidean)')
    parser.add_argument('--output-dir', type=str, default='knn_results',
                        help='Directory for saved plots')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random seed')
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.random_state)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("k-NN CLASSIFICATION")
    print("=" * 70)
    print(f"Dataset:      {args.data}")
    print(f"Test size:    {args.test_size}")
    print(f"k:            {args.n_neighbors}")
    print(f"Weights:      {args.weights}")
    print(f"p (distance): {args.p}")
    print("=" * 70)

    # Load & preprocess
    X, y = load_and_preprocess_data(args.data)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    # Train KNN
    knn = KNNClassifier(
        n_neighbors=args.n_neighbors,
        weights=args.weights,
        p=args.p
    )
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted', zero_division=0
    )

    print("\nRESULTS")
    print("-" * 70)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    class_names = ['No', '>30', '<30']
    cm_path = os.path.join(args.output_dir, 'knn_confusion_matrix.png')
    plot_confusion_matrix(cm, class_names, cm_path)

    # PCA visualization
    pca_path = os.path.join(args.output_dir, 'knn_pca_predictions.png')
    plot_pca_scatter(X_test, y_test, y_pred, pca_path)

    print("\nInterpretation notes:")
    print("  • Use confusion matrix + per-class F1 to discuss performance")
    print("  • Relate misclassification patterns to class imbalance and feature space\n")
    print("Results saved in:", args.output_dir)
    print("=" * 70)


if __name__ == '__main__':
    main()
