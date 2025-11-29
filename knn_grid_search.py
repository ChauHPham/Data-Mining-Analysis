"""
k-NN Grid Search for Diabetic Readmission
CMPT 459 Course Project

Performs manual grid search over KNN hyperparameters with K-fold CV
on a PCA-reduced feature space, plots accuracy vs k, and evaluates
the best model on a hold-out test set.
"""

import argparse
import os
from itertools import product

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

from knn_classifier import KNNClassifier  # uses the class defined in knn_classifier.py


# -----------------------------------------------------------
# Data loader (same as in knn_classifier.py for consistency)
# -----------------------------------------------------------
def load_and_preprocess_data(path: str):
    print("Loading data...")
    df = pd.read_csv(path)
    print(f"Original shape: {df.shape}")

    # Replace '?' with NaN
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

    # Drop obvious IDs
    for col in ["encounter_id", "patient_nbr"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Scale numeric
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    X = df.drop(columns=["readmitted"]).values
    y = df["readmitted"].values.astype(int)

    print("Preprocessing complete! Final shape:", X.shape)
    return X, y


# -----------------------------------------------------------
# Simple manual grid search with K-fold CV
# -----------------------------------------------------------
def kfold_split(n_samples: int, cv: int, shuffle: bool, random_state: int):
    idx = np.arange(n_samples)
    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(idx)
    folds = np.array_split(idx, cv)
    splits = []
    for i in range(cv):
        val_idx = folds[i]
        train_idx = np.concatenate(folds[:i] + folds[i+1:])
        splits.append((train_idx, val_idx))
    return splits


def grid_search_knn(X, y, ks, weights_list, p_values, cv=5, random_state=42):
    """
    Run grid search over KNN hyperparameters on (X, y).
    X is assumed to already be PCA-reduced.
    """
    n_samples = X.shape[0]
    splits = kfold_split(n_samples, cv=cv, shuffle=True, random_state=random_state)

    results = []
    best_score = -np.inf
    best_params = None

    print("\nRunning grid search over:")
    print("  ks       :", ks)
    print("  weights  :", weights_list)
    print("  p values :", p_values)
    print(f"  CV folds : {cv}")
    print("-" * 70)

    for k, w, p in product(ks, weights_list, p_values):
        cv_scores = []

        for train_idx, val_idx in splits:
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            clf = KNNClassifier(n_neighbors=k, weights=w, p=p)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_val)
            # y_pred should already be integers from KNNClassifier
            acc = accuracy_score(y_val, y_pred)
            cv_scores.append(acc)

        mean_acc = float(np.mean(cv_scores))
        results.append({
            "k": k,
            "weights": w,
            "p": p,
            "mean_acc": mean_acc,
            "cv_scores": cv_scores,
        })

        print(f"k={k:2d}, weights={w:8s}, p={p} -> mean CV acc={mean_acc:.4f}")

        if mean_acc > best_score:
            best_score = mean_acc
            best_params = {"n_neighbors": k, "weights": w, "p": p}

    print("-" * 70)
    print(f"Best params: {best_params}, mean CV accuracy = {best_score:.4f}")
    return results, best_params, best_score


def plot_grid_results(results, out_path):
    """
    Plot mean accuracy vs k, with separate lines per (weights, p) combination.
    """
    combos = {}
    for r in results:
        key = (r["weights"], r["p"])
        combos.setdefault(key, []).append((r["k"], r["mean_acc"]))

    plt.figure(figsize=(8, 6))
    for (w, p), kv_list in combos.items():
        kv_list = sorted(kv_list, key=lambda t: t[0])  # sort by k
        ks = [x[0] for x in kv_list]
        accs = [x[1] for x in kv_list]
        plt.plot(ks, accs, marker='o', label=f'w={w}, p={p}')

    plt.xlabel('k (number of neighbors)')
    plt.ylabel('Mean CV Accuracy')
    plt.title('k-NN Grid Search (CV Accuracy vs k)')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved grid search plot to {out_path}")


def parse_list_int(s: str):
    return [int(x) for x in s.split(',') if x.strip()]


def parse_list_str(s: str):
    return [x.strip() for x in s.split(',') if x.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description='k-NN Grid Search for Diabetic Readmission')
    parser.add_argument('--data', type=str, default='data/diabetic_data.csv',
                        help='Path to dataset CSV')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Test set proportion for final evaluation')
    parser.add_argument('--k-range', type=str, default='3,5,7',
                        help='Comma-separated k values, e.g. "3,5,7,9"')
    parser.add_argument('--weights', type=str, default='uniform,distance',
                        help='Comma-separated weights options, e.g. "uniform,distance"')
    parser.add_argument('--p-values', type=str, default='2',
                        help='Comma-separated Minkowski p values, e.g. "1,2"')
    parser.add_argument('--cv', type=int, default=3,
                        help='Number of CV folds')
    parser.add_argument('--pca-components', type=int, default=50,
                        help='Number of PCA components before grid search (default: 50)')
    parser.add_argument('--max-train-samples', type=int, default=50000,
                        help='Max training samples for grid search (for speed/memory)')
    parser.add_argument('--output-dir', type=str, default='knn_grid_results',
                        help='Directory for results & plots')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random seed')
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.random_state)
    os.makedirs(args.output_dir, exist_ok=True)

    ks = parse_list_int(args.k_range)
    weights_list = parse_list_str(args.weights)
    p_values = parse_list_int(args.p_values)

    print("=" * 70)
    print("k-NN GRID SEARCH")
    print("=" * 70)
    print(f"Dataset:         {args.data}")
    print(f"k values:        {ks}")
    print(f"Weights:         {weights_list}")
    print(f"p values:        {p_values}")
    print(f"CV folds:        {args.cv}")
    print(f"Test size:       {args.test_size}")
    print(f"PCA components:  {args.pca_components}")
    print(f"Max train size:  {args.max_train_samples}")
    print("=" * 70)

    # Load & preprocess (high-dimensional space)
    X, y = load_and_preprocess_data(args.data)

    # Hold-out test set for final evaluation
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=args.test_size,
        random_state=args.random_state, stratify=y
    )

    # Optional subsampling of training data (for speed)
    n_train_full = X_train_full.shape[0]
    if n_train_full > args.max_train_samples:
        print(f"Training set has {n_train_full} samples; sampling {args.max_train_samples} for grid search...")
        rng = np.random.default_rng(args.random_state)
        idx = rng.choice(n_train_full, size=args.max_train_samples, replace=False)
        X_train_gs = X_train_full[idx]
        y_train_gs = y_train_full[idx]
    else:
        X_train_gs = X_train_full
        y_train_gs = y_train_full

    # Apply PCA on training data (for grid search + final model)
    print("\nApplying PCA on training data...")
    pca = PCA(n_components=args.pca_components, random_state=args.random_state)
    X_train_gs_pca = pca.fit_transform(X_train_gs)
    explained = pca.explained_variance_ratio_.sum()
    print(f"PCA complete. Shape: {X_train_gs_pca.shape}, explained variance: {explained:.2%}")

    # Also transform full train + test sets with same PCA model
    X_train_full_pca = pca.transform(X_train_full)
    X_test_pca = pca.transform(X_test)

    # Grid search on PCA-reduced training subset
    results, best_params, best_cv_score = grid_search_knn(
        X_train_gs_pca, y_train_gs, ks, weights_list, p_values,
        cv=args.cv, random_state=args.random_state
    )

    # Save grid search results
    results_df = pd.DataFrame([
        {
            "k": r["k"],
            "weights": r["weights"],
            "p": r["p"],
            "mean_cv_acc": r["mean_acc"],
        }
        for r in results
    ])
    csv_path = os.path.join(args.output_dir, 'knn_grid_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved grid search CSV to {csv_path}")

    # Plot grid search curves
    plot_path = os.path.join(args.output_dir, 'knn_grid_search_plot.png')
    plot_grid_results(results, plot_path)

    # Train best model on full PCA-reduced training set & evaluate on test
    print("\nTraining best model on full PCA-reduced training set and evaluating on test set...")
    best_knn = KNNClassifier(
        n_neighbors=best_params["n_neighbors"],
        weights=best_params["weights"],
        p=best_params["p"],
    )
    best_knn.fit(X_train_full_pca, y_train_full.astype(int))
    y_pred_test = best_knn.predict(X_test_pca).astype(int)
    test_acc = accuracy_score(y_test.astype(int), y_pred_test)

    print("\nFINAL EVALUATION")
    print("-" * 70)
    print(f"Best params from CV: {best_params}")
    print(f"Mean CV accuracy:    {best_cv_score:.4f}")
    print(f"Test accuracy:       {test_acc:.4f}")
    print("-" * 70)
    print("Interpretation:")
    print("  • Use CV accuracy to justify chosen k / weights / distance metric.")
    print("  • Use hold-out test accuracy to estimate generalization performance.")
    print("  • Mention PCA + subsampling as practical choices for speed/memory.")
    print(f"\nAll results saved in: {args.output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
