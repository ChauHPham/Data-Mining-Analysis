"""
Manual Random Search for SVM Hyperparameters
CMPT 459 Course Project

Randomly samples hyperparameter combinations and evaluates each
via K-fold cross-validation on PCA-transformed data.
"""

import argparse
import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

from svm_classifier import SVMClassifier


# ============================================================
#                PREPROCESSING (same as before)
# ============================================================

def load_and_preprocess_data(path):
    print("Loading data...")
    df = pd.read_csv(path)
    df = df.replace("?", np.nan)

    # Drop columns with >40% missing
    threshold = 0.4 * len(df)
    df = df.dropna(thresh=threshold, axis=1)

    # Fill categorical missing
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna("Unknown")

    # Encode target
    df["readmitted"] = df["readmitted"].map({"NO": 0, ">30": 1, "<30": 2})

    # Encode categoricals
    cat_cols = df.select_dtypes(include="object").columns
    le = LabelEncoder()
    for col in cat_cols:
        if df[col].nunique() < 10:
            df[col] = le.fit_transform(df[col].astype(str))
        else:
            df = pd.get_dummies(df, columns=[col], drop_first=True)

    # Remove ID columns
    for col in ["encounter_id", "patient_nbr"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Scale numeric
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    df[num_cols] = StandardScaler().fit_transform(df[num_cols])

    X = df.drop(columns=["readmitted"]).values
    y = df["readmitted"].values.astype(int)

    print("Preprocessing complete. Final shape:", X.shape)
    return X, y


# ============================================================
#                   K-FOLD SPLITTING
# ============================================================

def kfold_split(n_samples, cv, shuffle=True, random_state=42):
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


# ============================================================
#                   RANDOM SEARCH IMPLEMENTATION
# ============================================================

def random_search_svm(X, y, search_space, n_iter=10, cv=3, random_state=42):
    rng = np.random.default_rng(random_state)
    splits = kfold_split(len(X), cv=cv, shuffle=True, random_state=random_state)

    results = []
    best_score = -np.inf
    best_params = None

    print("\nRandom search space:")
    for key, val in search_space.items():
        print(f"  {key}: {val}")
    print(f"\nSampling {n_iter} random configurations...\n")

    for it in range(n_iter):
        params = {
            "kernel": rng.choice(search_space["kernel"]),
            "C": float(rng.uniform(*search_space["C"])),
            "gamma": rng.choice(search_space["gamma"]),
            "degree": int(rng.choice(search_space["degree"])),
        }

        cv_scores = []

        # Perform CV
        for train_idx, val_idx in splits:
            clf = SVMClassifier(
                kernel=params["kernel"],
                C=params["C"],
                gamma=params["gamma"],
                degree=params["degree"],
                random_state=random_state,
            )
            clf.fit(X[train_idx], y[train_idx])
            y_pred = clf.predict(X[val_idx])
            cv_scores.append(accuracy_score(y[val_idx], y_pred))

        mean_acc = float(np.mean(cv_scores))

        results.append({
            "kernel": params["kernel"],
            "C": params["C"],
            "gamma": params["gamma"],
            "degree": params["degree"],
            "mean_acc": mean_acc,
        })

        print(f"[{it+1:02d}/{n_iter}] {params} → mean CV acc = {mean_acc:.4f}")

        if mean_acc > best_score:
            best_score = mean_acc
            best_params = params

    print("\nBest params:", best_params)
    print("Best CV accuracy:", best_score)
    return results, best_params, best_score


# ============================================================
#                         ARGPARSE
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Manual Random Search for SVM")

    parser.add_argument("--data", type=str, default="data/diabetic_data.csv")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--cv", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--pca-components", type=int, default=50)
    parser.add_argument("--max-train-samples", type=int, default=30000)
    parser.add_argument("--output-dir", type=str, default="svm_random_results")
    parser.add_argument("--random-state", type=int, default=42)

    # Kernel override
    parser.add_argument("--kernel-only", type=str, default=None,
                        help="Force random search to use only this kernel.")

    # Fast mode
    parser.add_argument("--fast-mode", action="store_true",
                        help="Enable ultra-fast debug mode.")

    return parser.parse_args()


# ============================================================
#                            MAIN
# ============================================================

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.random_state)

    print("=" * 70)
    print("          MANUAL RANDOM SEARCH — SVM CLASSIFIER")
    print("=" * 70)

    # FAST MODE OVERRIDES
    if args.fast_mode:
        print("\n⚡ FAST MODE ENABLED — drastically reducing runtime.\n")
        args.pca_components = 10
        args.max_train_samples = 4000
        args.cv = 1
        args.iterations = 2
        args.kernel_only = "linear"

    # Load data
    X, y = load_and_preprocess_data(args.data)

    # Train/test split
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=args.test_size,
        random_state=args.random_state, stratify=y
    )

    # Optional subsampling
    if len(X_train_full) > args.max_train_samples:
        rng = np.random.default_rng(args.random_state)
        idx = rng.choice(len(X_train_full), args.max_train_samples, replace=False)
        X_train_gs = X_train_full[idx]
        y_train_gs = y_train_full[idx]
    else:
        X_train_gs = X_train_full
        y_train_gs = y_train_full

    # PCA
    print(f"\nApplying PCA ({args.pca_components} components)...")
    pca = PCA(n_components=args.pca_components, random_state=args.random_state)
    X_train_gs_pca = pca.fit_transform(X_train_gs)
    X_train_full_pca = pca.transform(X_train_full)
    X_test_pca = pca.transform(X_test)
    print("PCA complete.")

    # Define random search space
    search_space = {
        "kernel": ["linear", "rbf", "sigmoid", "poly"],
        "C": (0.01, 10),
        "gamma": ["scale", "auto"],
        "degree": [2, 3, 4],
    }

    # Apply kernel override
    if args.kernel_only is not None:
        print(f"\n➡️ Kernel override active — using only: {args.kernel_only}")
        search_space["kernel"] = [args.kernel_only]

    # Run random search
    results, best_params, best_cv_score = random_search_svm(
        X_train_gs_pca, y_train_gs,
        search_space,
        n_iter=args.iterations,
        cv=args.cv,
        random_state=args.random_state
    )

    # Save CSV
    results_df = pd.DataFrame(results)
    csv_path = os.path.join(args.output_dir, "svm_random_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved results to {csv_path}")

    # Train best model on FULL training set
    print("\nTraining best model on full PCA-reduced training set...")
    best_clf = SVMClassifier(**best_params)
    best_clf.fit(X_train_full_pca, y_train_full)
    y_pred_test = best_clf.predict(X_test_pca)

    test_acc = accuracy_score(y_test, y_pred_test)

    print("\nFINAL TEST PERFORMANCE")
    print("-" * 60)
    print("Best params:", best_params)
    print(f"Mean CV accuracy: {best_cv_score:.4f}")
    print(f"Test accuracy:    {test_acc:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
