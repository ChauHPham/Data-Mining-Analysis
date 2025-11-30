"""
k-NN Classification for Diabetic Readmission
CMPT 459 Course Project

Trains a KNN classifier using PCA-reduced data (dim=50) with batched
distance computation to avoid memory errors. Evaluates accuracy,
precision, recall, F1, saves confusion matrix and PCA prediction plot.
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


# ============================================================
#                   KNN CLASSIFIER (Batched)
# ============================================================

class KNNClassifier:
    """
    Memory-safe KNN classifier using PCA + batched Minkowski distances.
    """

    def __init__(self, n_neighbors=5, weights="uniform", p=2, batch_size=500):
        self.n_neighbors = n_neighbors
        self.weights = weights  # "uniform" or "distance"
        self.p = p              # 1=Manhattan, 2=Euclidean
        self.batch_size = batch_size
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y.astype(int)
        return self

    # ----------------- BATCHED DISTANCE MATRIX -----------------

    def _compute_distances_batched(self, X):
        """
        Computes full distance matrix in batches.
        Avoids allocating huge arrays that cause MemoryError.
        """
        n_test = X.shape[0]
        n_train = self.X_train.shape[0]
        dists = np.zeros((n_test, n_train), dtype=np.float32)

        for start in range(0, n_test, self.batch_size):
            end = min(start + self.batch_size, n_test)
            X_batch = X[start:end]

            if self.p == 2:
                X_sq = np.sum(X_batch ** 2, axis=1).reshape(-1, 1)
                Xtr_sq = np.sum(self.X_train ** 2, axis=1)
                dist_sq = X_sq + Xtr_sq - 2 * X_batch @ self.X_train.T
                dist_sq = np.maximum(dist_sq, 0.0)
                dists[start:end] = np.sqrt(dist_sq, dtype=np.float32)

            else:
                d = np.sum(np.abs(X_batch[:, None, :] - self.X_train[None, :, :]), axis=2)
                dists[start:end] = d.astype(np.float32)

        return dists

    # ----------------- PREDICTION FOR ONE SAMPLE -----------------

    def _predict_one(self, dist_row):
        """
        Determine the class of a single test sample using nearest neighbors.
        """
        k = self.n_neighbors
        nn_idx = np.argpartition(dist_row, k)[:k]
        nn_labels = self.y_train[nn_idx]
        nn_dists = dist_row[nn_idx]

        if self.weights == "uniform":
            return int(Counter(nn_labels).most_common(1)[0][0])

        else:
            eps = 1e-8
            w = 1.0 / (nn_dists + eps)
            scores = {}
            for lbl, weight in zip(nn_labels, w):
                scores[lbl] = scores.get(lbl, 0.0) + weight
            return int(max(scores.items(), key=lambda kv: kv[1])[0])

    # ----------------- BATCH PREDICTION -----------------

    def predict(self, X):
        """
        Return integer predictions for all test samples.
        """
        distances = self._compute_distances_batched(X)
        preds = np.apply_along_axis(self._predict_one, 1, distances)
        return preds.astype(int)


# ============================================================
#                   PREPROCESSING
# ============================================================

def load_and_preprocess_data(path):
    print("Loading data...")
    df = pd.read_csv(path)
    print(f"Original shape: {df.shape}")

    df = df.replace("?", np.nan)

    # Drop columns with >40% missing
    threshold = 0.4 * len(df)
    df = df.dropna(thresh=threshold, axis=1)

    # Fill categorical NA
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna("Unknown")

    # Encode target
    df["readmitted"] = df["readmitted"].map({"NO": 0, ">30": 1, "<30": 2})

    # Encode categorical variables
    cat_cols = df.select_dtypes(include="object").columns
    le = LabelEncoder()
    for col in cat_cols:
        if df[col].nunique() < 10:
            df[col] = le.fit_transform(df[col].astype(str))
        else:
            df = pd.get_dummies(df, columns=[col], drop_first=True)

    # Remove IDs if present
    for col in ["encounter_id", "patient_nbr"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Scale numerical features
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    X = df.drop(columns=["readmitted"]).values
    y = df["readmitted"].values.astype(int)

    print("Preprocessing complete! Final shape:", X.shape)
    return X, y


# ============================================================
#                   VISUALIZATION HELPERS
# ============================================================

def plot_confusion_matrix(cm, class_names, out_path):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title("k-NN Confusion Matrix")
    plt.colorbar()

    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names)
    plt.yticks(ticks, class_names)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]),
                     ha="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix to {out_path}")


def plot_pca_scatter(X, y_true, y_pred, out_path):
    print("Running PCA for visualization...")
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)

    correct = (y_true == y_pred)

    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[correct, 0], X_pca[correct, 1],
                c=y_pred[correct], cmap="viridis",
                s=20, alpha=0.5, label="Correct")

    plt.scatter(X_pca[~correct, 0], X_pca[~correct, 1],
                c=y_pred[~correct], cmap="viridis",
                s=60, alpha=0.9, marker="x", label="Incorrect", edgecolors="black")

    plt.title("k-NN Predictions in PCA Space (colored by predicted class)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved PCA prediction scatter to {out_path}")


# ============================================================
#                   ARGPARSE
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="k-NN Classification with PCA + batching")
    parser.add_argument("--data", type=str, default="data/diabetic_data.csv",
                        help="Path to CSV")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--n-neighbors", type=int, default=5)
    parser.add_argument("--weights", type=str, default="uniform",
                        choices=["uniform", "distance"])
    parser.add_argument("--p", type=int, default=2, choices=[1, 2])
    parser.add_argument("--output-dir", type=str, default="knn_results")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


# ============================================================
#                   MAIN
# ============================================================

def main():
    args = parse_args()
    np.random.seed(args.random_state)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("k-NN CLASSIFICATION (PCA 50 + Batched KNN)")
    print("=" * 70)

    # Load data
    X, y = load_and_preprocess_data(args.data)

    # Apply PCA **before** train/test split (important!)
    print("Applying PCA (50 components)...")
    pca = PCA(n_components=50, random_state=42)
    X_pca = pca.fit_transform(X)
    print(f"PCA complete. Shape: {X_pca.shape}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    # Train classifier
    knn = KNNClassifier(
        n_neighbors=args.n_neighbors,
        weights=args.weights,
        p=args.p,
        batch_size=500
    )

    print("Training k-NN...")
    knn.fit(X_train, y_train)

    print("Predicting...")
    y_pred = knn.predict(X_test).astype(int)

    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )

    print("\nRESULTS")
    print("-" * 70)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    cm_path = os.path.join(args.output_dir, "knn_confusion_matrix.png")
    plot_confusion_matrix(cm, ["No", ">30", "<30"], cm_path)

    pca_path = os.path.join(args.output_dir, "knn_pca_predictions.png")
    plot_pca_scatter(X_test, y_test, y_pred, pca_path)

    print("\n✓ Results saved in:", args.output_dir)
    print("=" * 70)


if __name__ == "__main__":
    main()
