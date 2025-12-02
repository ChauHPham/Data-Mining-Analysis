"""
DBScan Clustering Analysis for Diabetic Patient Data
CMPT 459 Course Project

This script performs density-based clustering on the diabetic dataset,
evaluates cluster quality using silhouette scores, and visualizes the results.

Adjusted to match the code structure of cluster_analysis.py (KMeans) and hierarchical_cluster_analysis.py (Hierchical).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import argparse

from dbscan_clustering import DBScan



# Parse Args from Command Line 
def parse_args():
    parser = argparse.ArgumentParser(description = 'DBScan Clustering for Diabetic Data')
    parser.add_argument('--data', type = str, default = 'data/diabetic_data.csv',
                        help = 'Path to diabetic dataset')
    parser.add_argument('--pca-components', type = int, default=50,
                        help = 'Number of PCA components before clustering')
    parser.add_argument('--vis-dims', type = int, default = 2, choices = [2, 3],
                        help = 'Dimensions for visualization (2 or 3)')
    parser.add_argument('--sample-size', type = int, default = 1000,
                        help = 'Limit number of samples (DBScan runs over every sample point)')
    parser.add_argument('--random-seed', type = int, default = 42,
                        help = 'Random seed')

    return parser.parse_args()

# Load + Preprocess Dataset (same as cluster_analysis.py)
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

# PCA helper
def apply_pca(X, n_components):
    """ Apply PCA for dimensional reduction of dataset."""
    print(f"Running PCA with {n_components} components...")
    pca = PCA(n_components = n_components)
    X_pca = pca.fit_transform(X)
    print("PCA done running. Explained variance:", np.sum(pca.explained_variance_ratio_))
    return X_pca

# Scatter Plot Visualization (same structure as cluster_analysis.py and hierarchical_cluster_analysis.py)
def visualize_cluster(X_vis, clustering, title, save_path, silhouette_score = None):
    """ Visualization of 2D or 3D scatterplots with colour-coded clusters."""
    num_dims = X_vis.shape[1]
    # Get unique clusters 
    unique_clusters = np.unique(clustering)
    num_clusters = len(unique_clusters)

    # Colour clusters 
    if hasattr(matplotlib, "colormaps"):
        cmap = matplotlib.colormaps["tab20"]
    else:
        cmap = plt.cm.get_cmap("tab20")


    if num_dims == 2: #2D Visualization
        plt.figure(figsize = (10, 8))
        for i, cid in enumerate(unique_clusters):
            mask = clustering == cid
            c_value = i / max(num_clusters - 1, 1) if num_clusters > 1 else 0
            color = cmap(c_value)
            plt.scatter(X_vis[mask, 0], X_vis[mask, 1],
                        s = 20, alpha = 0.6, color = color, edgecolors = "black",
                        linewidths = 0.5, label = f"Cluster {cid}")

        # Add sihouette score to title 
        if silhouette_score is not None:
            title += f"\nSilhouette: {silhouette_score:.4f}"

        plt.title(title, fontsize = 14, fontweight = "bold")
        plt.xlabel("Principal Component 1", fontsize=12)
        plt.ylabel("Principal Component 2", fontsize=12)
        plt.legend(bbox_to_anchor = (1.05, 1), loc = "upper left")
        plt.tight_layout()
        plt.savefig(save_path, dpi = 300, bbox_inches = "tight")
        plt.close()

    elif num_dims == 3: #3D Visualization
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize = (12, 9))
        ax = fig.add_subplot(111, projection = '3d')

        for i, cid in enumerate(unique_clusters):
            mask = clustering == cid
            c_value = i / max(num_clusters - 1, 1) if num_clusters > 1 else 0
            color = cmap(c_value)
            ax.scatter(X_vis[mask, 0], X_vis[mask, 1], X_vis[mask, 2],
                       s = 20, alpha = 0.6, color = color, edgecolors = "black",
                       linewidths = 0.5, label = f"Cluster {int(cid)}")

        # Add sihouette score to title
        if silhouette_score is not None:
            title += f"\nSilhouette: {silhouette_score:.4f}"

        ax.set_title(title, fontsize = 14, fontweight = "bold")
        ax.set_xlabel("Principal Component 1", fontsize = 11)
        ax.set_ylabel("Principal Component 2", fontsize = 11)
        ax.set_zlabel("Principal Component 3", fontsize = 11)
        ax.legend(bbox_to_anchor = (1.05, 1), loc = "upper left")

        plt.savefig(save_path, dpi = 300, bbox_inches = "tight")
        plt.close()

# Main script
def main():
    """ Performs DBScan clustering. """
    args = parse_args()
    np.random.seed(args.random_seed)

    # Load and preprocess data 
    X, target = load_and_preprocess_data(args.data)

    # Take sample of data (same as hierarchical_cluster_analysis.py)
    if len(X) > args.sample_size:
        print(f"Sampling {args.sample_size} points for DBScan...")
        idx = np.random.choice(len(X), args.sample_size, replace = False)
        X = X[idx]
        target = target.iloc[idx].values
    else:
        target = target.values
        
    # Apply PCA to reduce dimensions of data
    X_pca = apply_pca(X, args.pca_components)

    silhouettes = []
    best_score = -1
    best_clustering = None

    print("Cluster Analysis: Testing DBScan Clustering")

    # Performs DBScan on sample data points
    db = DBScan(0.2, 5) # choice of epsilon and minimum points in neighbourhood
    clustering = db.fit(X_pca)

    # Get silhouette score (only if we have more than 1 cluster and at least 2 samples per cluster)
    unique_labels = np.unique(clustering)
    if len(unique_labels) > 1 and all(np.sum(clustering == label) > 1 for label in unique_labels if label != 0):
        sil = silhouette_score(X_pca, clustering)
    else:
        sil = -1
        print("Warning: Cannot compute silhouette score - need at least 2 clusters with 2+ samples each")
    silhouettes.append(sil)
    print(f"Silhouette = {sil:.4f}")

    if sil > best_score:
        best_score = sil
        best_clustering = clustering.copy()

    # Print results summary
    print("\nSummary of Results:")
    for idx, sil in enumerate(silhouettes):
        marker = " <-- Best" if idx == 0 else ""
        print(f"Silhouette = {sil:.4f}{marker}")


    print("Generating visualizations...")

    print("Generating Silhouette Score Plot...")
    plt.figure(figsize = (10, 6))
    plt.plot(silhouettes, "o-", color = "steelblue")
    plt.title("DBScan Clustering Silhouette Scores")
    plt.xlabel("Epsilon Values")
    plt.ylabel("Silhouette Coefficient")
    plt.grid(alpha = 0.3)
    plt.savefig("dbscan_silhouette_scores.png", dpi = 300, bbox_inches = "tight")
    plt.close()
    print("\nSaved as dbscan_silhouette_scores.png.")

    # Reduce dataset for 2D and 3D Visualization 
    print(f"\nReducing to {args.vis_dims}D for visualization...")
    X_vis = PCA(n_components=args.vis_dims).fit_transform(X)

    print(f"\nVisualizing clusters...")
    vis_title = f"DBScan Clustering"
    vis_file = f"dbscan.png"

    # Produce scatter plot for clustering
    # visualize_cluster(X_vis, best_clustering, vis_title, vis_file, silhouette=best_score)
    visualize_cluster(X_vis, best_clustering, vis_title, vis_file)
    print(f"Saved scatterplot as {vis_file}.")
    print("DBScan clustering analysis complete!")
    print("\n Files saved:")
    print("- dbscan_silhouette_scores.png.")
    print(f"- {vis_file}")

if __name__ == "__main__":
    main()
