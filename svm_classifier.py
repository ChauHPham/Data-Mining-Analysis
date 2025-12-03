"""
SVM Classification for Diabetic Readmission
CMPT 459 Course Project

Trains an SVM classifier using PCA-reduced data (dim=50).
Evaluates accuracy, precision, recall, F1, saves confusion
matrix and PCA prediction plot.
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
from sklearn.svm import SVC


# ============================================================
#                   SVM CLASSIFIER
# ============================================================

class SVMClassifier:
    """
    Simple SVM wrapper for unified interface with KNNClassifier.
    """

    def __init__(self, kernel="rbf", C=1.0, gamma="scale", degree=3, random_state=42):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.random_state = random_state
        self.model = None

    def fit(self, X, y):
        """
        Fit the SVM classifier.
        """
        self.model = SVC(
            kernel=self.kernel,
            C=self.C,
            gamma=self.gamma,
            degree=self.degree,
            probability=False,
            random_state=self.random_state,
        )
        self.model.fit(X, y)
        return self

    def predict(self, X):
        """
        Predict class labels.
        """
        if self.model is None:
            raise ValueError("SVMClassifier must be fitted before calling predict().")
        return self.model.predict(X).astype(int)


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
    plt.title("SVM Confusion Matrix")
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
    plt.scatter(
        X_pca[correct, 0], X_pca[correct, 1],
        c=y_pred[correct], cmap="viridis",
        s=20, alpha=0.5, label="Correct"
    )

    plt.scatter(
        X_pca[~correct, 0], X_pca[~correct, 1],
        c=y_pred[~correct], cmap="viridis",
        s=60, alpha=0.9, marker="x",
        edgecolors="black", label="Incorrect"
    )

    plt.title("SVM Predictions in PCA Space (colored by predicted class)")
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
    parser = argparse.ArgumentParser(description="SVM Classification with PCA")
    parser.add_argument("--data", type=str, default="data/diabetic_data.csv")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--pca-components", type=int, default=50)

    # SVM hyperparameters
    parser.add_argument("--kernel", type=str, default="rbf",
                        choices=["linear", "rbf", "poly", "sigmoid"])
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--gamma", type=str, default="scale")
    parser.add_argument("--degree", type=int, default=3)

    parser.add_argument("--output-dir", type=str, default="svm_results")
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
    print(f"SVM CLASSIFICATION (PCA {args.pca_components} components)")
    print("=" * 70)

    # Load data
    X, y = load_and_preprocess_data(args.data)

    # PCA before train/test split (consistent with your KNN pipeline)
    print(f"Applying PCA ({args.pca_components} components)...")
    pca = PCA(n_components=args.pca_components, random_state=42)
    X_pca = pca.fit_transform(X)
    print(f"PCA complete. Shape: {X_pca.shape}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y
    )

    # Train classifier
    svm = SVMClassifier(
        kernel=args.kernel,
        C=args.C,
        gamma=args.gamma,
        degree=args.degree,
        random_state=args.random_state,
    )

    print("Training SVM...")
    svm.fit(X_train, y_train)

    print("Predicting...")
    y_pred = svm.predict(X_test)

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
    cm_path = os.path.join(args.output_dir, "svm_confusion_matrix.png")
    plot_confusion_matrix(cm, ["No", ">30", "<30"], cm_path)

    pca_path = os.path.join(args.output_dir, "svm_pca_predictions.png")
    plot_pca_scatter(X_test, y_test, y_pred, pca_path)

    print("\n✓ Results saved in:", args.output_dir)
    print("=" * 70)


if __name__ == "__main__":
    main()
