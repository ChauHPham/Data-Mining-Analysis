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
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors
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
        
        # Plot noise points first (if any)
        if 0 in unique_clusters:
            noise_mask = clustering == 0
            plt.scatter(X_vis[noise_mask, 0], X_vis[noise_mask, 1],
                        s = 30, alpha = 0.5, color = "gray", 
                        marker = "x", linewidths = 0.5, label = "Noise")
        
        # Plot actual clusters (excluding noise)
        cluster_ids = [cid for cid in unique_clusters if cid != 0]
        for i, cid in enumerate(cluster_ids):
            mask = clustering == cid
            c_value = i / max(len(cluster_ids) - 1, 1) if len(cluster_ids) > 1 else 0
            color = cmap(c_value)
            plt.scatter(X_vis[mask, 0], X_vis[mask, 1],
                        s = 20, alpha = 0.6, color = color, edgecolors = "black",
                        linewidths = 0.5, label = f"Cluster {int(cid)}")

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

        # Plot noise points first (if any)
        if 0 in unique_clusters:
            noise_mask = clustering == 0
            ax.scatter(X_vis[noise_mask, 0], X_vis[noise_mask, 1], X_vis[noise_mask, 2],
                       s = 30, alpha = 0.5, color = "gray",
                       marker = "x", linewidths = 0.5, label = "Noise")
        
        # Plot actual clusters (excluding noise)
        cluster_ids = [cid for cid in unique_clusters if cid != 0]
        for i, cid in enumerate(cluster_ids):
            mask = clustering == cid
            c_value = i / max(len(cluster_ids) - 1, 1) if len(cluster_ids) > 1 else 0
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

    # Find appropriate epsilon using k-distance graph
    print("\nFinding appropriate epsilon using k-distance graph...")
    minPts = 5
    neighbors = NearestNeighbors(n_neighbors=minPts)
    neighbors_fit = neighbors.fit(X_pca)
    distances, indices = neighbors_fit.kneighbors(X_pca)
    
    # Sort distances to k-th nearest neighbor
    k_distances = np.sort(distances[:, minPts-1])
    
    # Use percentiles to suggest epsilon values
    eps_25 = np.percentile(k_distances, 25)
    eps_50 = np.percentile(k_distances, 50)
    eps_75 = np.percentile(k_distances, 75)
    eps_90 = np.percentile(k_distances, 90)
    
    print(f"k-distance statistics (k={minPts}):")
    print(f"  25th percentile: {eps_25:.4f}")
    print(f"  50th percentile: {eps_50:.4f}")
    print(f"  75th percentile: {eps_75:.4f}")
    print(f"  90th percentile: {eps_90:.4f}")
    
    # Test multiple epsilon values around the suggested range
    eps_values = [eps_25, eps_50, eps_75, eps_90, eps_90 * 1.5, eps_90 * 2.0]
    
    silhouettes = []
    calinski_harabasz_scores = []
    davies_bouldin_scores = []
    eps_results = []
    best_score = -1
    best_clustering = None
    best_eps = None

    print("\nCluster Analysis: Testing DBScan Clustering with different epsilon values")
    print("=" * 80)

    for eps in eps_values:
        db = DBScan(eps, minPts)
        clustering = db.fit(X_pca)
        
        unique_labels = np.unique(clustering)
        n_clusters = len(unique_labels[unique_labels != 0])
        n_noise = np.sum(clustering == 0)
        
        # Get evaluation metrics (only if we have more than 1 cluster)
        if n_clusters > 1:
            # Filter to only non-noise points
            mask = clustering != 0
            if np.sum(mask) > 1:
                # Check if all clusters have at least 2 points
                valid_clusters = [l for l in unique_labels if l != 0 and np.sum(clustering == l) >= 2]
                if len(valid_clusters) > 1:
                    mask = np.isin(clustering, valid_clusters)
                    sil = silhouette_score(X_pca[mask], clustering[mask])
                    ch_score = calinski_harabasz_score(X_pca[mask], clustering[mask])
                    db_score = davies_bouldin_score(X_pca[mask], clustering[mask])
                else:
                    sil = -1
                    ch_score = -1
                    db_score = -1
            else:
                sil = -1
                ch_score = -1
                db_score = -1
        else:
            sil = -1
            ch_score = -1
            db_score = -1
        
        silhouettes.append(sil)
        calinski_harabasz_scores.append(ch_score)
        davies_bouldin_scores.append(db_score)
        eps_results.append({'eps': eps, 'n_clusters': n_clusters, 'n_noise': n_noise, 
                           'silhouette': sil, 'calinski_harabasz': ch_score, 'davies_bouldin': db_score})
        
        status = "✓" if n_clusters > 0 else "✗"
        if sil > -1:
            print(f"{status} eps={eps:6.4f}: {n_clusters:2d} clusters, {n_noise:4d} noise, "
                  f"Silhouette={sil:7.4f}, CH={ch_score:7.4f}, DB={db_score:7.4f}")
        else:
            print(f"{status} eps={eps:6.4f}: {n_clusters:2d} clusters, {n_noise:4d} noise, "
                  f"Silhouette={sil:7.4f}")
        
        if sil > best_score and n_clusters > 0:
            best_score = sil
            best_clustering = clustering.copy()
            best_eps = eps

    print("=" * 80)
    
    # If no valid clustering found, use the one with most clusters
    if best_clustering is None:
        print("\nWarning: No clustering with valid silhouette score found!")
        print("Using clustering with most clusters...")
        max_clusters_idx = max(range(len(eps_results)), key=lambda i: eps_results[i]['n_clusters'])
        best_eps = eps_values[max_clusters_idx]
        db = DBScan(best_eps, minPts)
        best_clustering = db.fit(X_pca)
        best_score = silhouettes[max_clusters_idx]
    
    unique_labels = np.unique(best_clustering)
    n_clusters = len(unique_labels[unique_labels != 0])
    n_noise = np.sum(best_clustering == 0)
    
    print(f"\nBest result: eps={best_eps:.4f}")
    print(f"  Clusters: {n_clusters}")
    print(f"  Noise points: {n_noise}")
    if best_score > -1:
        best_idx = eps_values.index(best_eps)
        print(f"  Silhouette score: {best_score:.4f}")
        print(f"  Calinski-Harabasz: {calinski_harabasz_scores[best_idx]:.4f}")
        print(f"  Davies-Bouldin: {davies_bouldin_scores[best_idx]:.4f}")


    print("\nGenerating visualizations...")

    print("Generating Silhouette Score Plot...")
    plt.figure(figsize = (12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(k_distances, linewidth=1)
    plt.axhline(y=best_eps, color='red', linestyle='--', linewidth=2, label=f'Selected ε={best_eps:.4f}')
    plt.title("k-Distance Graph", fontsize=12, fontweight='bold')
    plt.xlabel("Points sorted by distance", fontsize=10)
    plt.ylabel(f"Distance to {minPts}-th nearest neighbor", fontsize=10)
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 2, 2)
    eps_labels = [f"{eps:.3f}" for eps in eps_values]
    plt.plot(range(len(eps_values)), silhouettes, "o-", color = "steelblue", linewidth=2, markersize=8)
    if best_eps is not None:
        best_idx = eps_values.index(best_eps)
        plt.axvline(x=best_idx, color='red', linestyle='--', linewidth=2, alpha=0.7)
    plt.xticks(range(len(eps_values)), eps_labels, rotation=45)
    plt.title("Silhouette Scores vs Epsilon", fontsize=12, fontweight='bold')
    plt.xlabel("Epsilon (ε) Value", fontsize=10)
    plt.ylabel("Silhouette Coefficient", fontsize=10)
    plt.grid(alpha = 0.3)
    
    plt.tight_layout()
    plt.savefig("dbscan_silhouette_scores.png", dpi = 300, bbox_inches = "tight")
    plt.close()
    print("Saved as dbscan_silhouette_scores.png.")

    # Reduce dataset for 2D and 3D Visualization 
    print(f"\nReducing to {args.vis_dims}D for visualization...")
    X_vis = PCA(n_components=args.vis_dims).fit_transform(X)

    print(f"\nVisualizing clusters...")
    vis_title = f"DBScan Clustering (eps={best_eps:.4f}, minPts={minPts})"
    vis_file = f"dbscan.png"

    # Produce scatter plot for clustering
    visualize_cluster(X_vis, best_clustering, vis_title, vis_file, silhouette_score=best_score if best_score > -1 else None)
    print(f"Saved scatterplot as {vis_file}.")
    print("DBScan clustering analysis complete!")
    print("\n Files saved:")
    print("- dbscan_silhouette_scores.png.")
    print(f"- {vis_file}")

if __name__ == "__main__":
    main()
