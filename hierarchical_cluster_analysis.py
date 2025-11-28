"""
Hierarchical Clustering Analysis for Diabetic Patient Data
CMPT 459 Course Project

This script performs Agglomerative Hierarchical Clustering on the diabetic dataset,
evaluates cluster quality using silhouette scores, and visualizes the results.

Designed to match the structure/style of cluster_analysis.py (KMeans).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import argparse

from hierarchical_clustering import HierarchicalClustering


# =========================================================
# 1. Parse Args
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(description='Hierarchical Clustering for Diabetic Data')
    parser.add_argument('--data', type=str, default='data/diabetic_data.csv',
                        help='Path to diabetic dataset')

    parser.add_argument('--min-k', type=int, default=2,
                        help='Minimum number of clusters to test')

    parser.add_argument('--max-k', type=int, default=6,
                        help='Maximum number of clusters to test')

    parser.add_argument('--pca-components', type=int, default=50,
                        help='Number of PCA components before clustering')

    parser.add_argument('--vis-dims', type=int, default=2, choices=[2, 3],
                        help='Dimensions for visualization (2 or 3)')

    parser.add_argument('--sample-size', type=int, default=2000,
                        help='Limit samples (Hierarchical Clustering is O(n^3))')

    parser.add_argument('--random-seed', type=int, default=42,
                        help='Random seed')

    return parser.parse_args()


# =========================================================
# 2. Load + Preprocess (same as your friend's code)
# =========================================================
def load_and_preprocess_data(path):
    print("Loading data...")
    df = pd.read_csv(path)
    print(f"Original shape: {df.shape}")

    # Replace '?'
    df = df.replace('?', np.nan)

    # Drop >40% missing
    threshold = 0.4 * len(df)
    df = df.dropna(thresh=threshold, axis=1)

    # Fill remaining categorical NA
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna("Unknown")

    # Encode target
    df["readmitted"] = df["readmitted"].map({'NO':0, '>30':1, '<30':2})

    # Encode categorical
    cat_cols = df.select_dtypes(include='object').columns
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

    # Normalize numeric
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    print("Preprocessing complete!")
    print("Final shape:", df.shape)

    target = df["readmitted"].copy()
    X = df.drop(columns=["readmitted"]).values

    return X, target


# =========================================================
# 3. PCA helper
# =========================================================
def apply_pca(X, n_components):
    print(f"Running PCA → {n_components} components...")
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    print("  Explained variance:", np.sum(pca.explained_variance_ratio_))
    return X_pca


# =========================================================
# 4. Visualization (same style as friend)
# =========================================================
def visualize_cluster(X_vis, labels, title, save_path, silhouette=None):
    dims = X_vis.shape[1]

    if hasattr(matplotlib, "colormaps"):
        cmap = matplotlib.colormaps["tab20"]
    else:
        cmap = plt.cm.get_cmap("tab20")

    unique = np.unique(labels)

    if dims == 2:
        plt.figure(figsize=(10, 8))
        for i, cluster in enumerate(unique):
            mask = labels == cluster
            color = cmap(i / max(len(unique)-1, 1))
            plt.scatter(X_vis[mask, 0], X_vis[mask, 1],
                        s=20, alpha=0.6, color=color, edgecolors="black",
                        label=f"Cluster {cluster}")

        if silhouette is not None:
            title += f"\nSilhouette: {silhouette:.4f}"

        plt.title(title, fontsize=14, fontweight="bold")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    elif dims == 3:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')

        for i, cluster in enumerate(unique):
            mask = labels == cluster
            color = cmap(i / max(len(unique)-1, 1))
            ax.scatter(X_vis[mask, 0], X_vis[mask, 1], X_vis[mask, 2],
                       s=20, alpha=0.6, color=color, edgecolors="black",
                       label=f"Cluster {cluster}")

        if silhouette is not None:
            title += f"\nSilhouette: {silhouette:.4f}"

        ax.set_title(title)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()


# =========================================================
# 5. MAIN SCRIPT
# =========================================================
def main():
    args = parse_args()
    np.random.seed(args.random_seed)

    # Load + preprocess
    X, target = load_and_preprocess_data(args.data)

    # Sample (hierarchical is heavy)
    if len(X) > args.sample_size:
        print(f"Sampling {args.sample_size} points (Hierarchical is expensive)...")
        idx = np.random.choice(len(X), args.sample_size, replace=False)
        X = X[idx]
        target = target.iloc[idx].values
    else:
        target = target.values

    # PCA for clustering
    X_pca = apply_pca(X, args.pca_components)

    ks = list(range(args.min_k, args.max_k+1))
    silhouettes = []

    best_k = None
    best_score = -1
    best_labels = None

    print("=== Testing Hierarchical Clustering for k ∈", ks, "===")

    for k in ks:
        print(f"\nClustering k={k}...", end=" ", flush=True)

        hc = HierarchicalClustering(n_clusters=k)
        labels = hc.fit(X_pca)

        sil = silhouette_score(X_pca, labels)
        silhouettes.append(sil)

        print(f"Silhouette = {sil:.4f}")

        if sil > best_score:
            best_score = sil
            best_k = k
            best_labels = labels.copy()

    # Print summary
    print("\n=== SUMMARY ===")
    for k, sil in zip(ks, silhouettes):
        marker = " <-- Best" if k == best_k else ""
        print(f"k={k}: Silhouette={sil:.4f}{marker}")

    # Silhouette plot
    plt.figure(figsize=(10, 6))
    plt.plot(ks, silhouettes, "o-", color="steelblue")
    plt.title("Hierarchical Clustering Silhouette Scores")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.grid(alpha=0.3)
    plt.xticks(ks)
    plt.savefig("hierarchical_silhouette_scores.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("\nSaved: hierarchical_silhouette_scores.png")

    # Visualization (2D/3D)
    print(f"\nReducing to {args.vis_dims}D for visualization...")
    X_vis = PCA(n_components=args.vis_dims).fit_transform(X)

    vis_title = f"Hierarchical Clustering (k={best_k})"
    vis_file = f"hierarchical_k{best_k}.png"

    visualize_cluster(X_vis, best_labels, vis_title, vis_file, silhouette=best_score)

    print(f"Saved: {vis_file}")
    print("\nHierarchical clustering complete!")


if __name__ == "__main__":
    main()
