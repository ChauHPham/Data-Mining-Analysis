"""
Cluster Analysis for Diabetic Patient Data
CMPT 459 Course Project

This script performs K-Means clustering on the diabetic patient dataset,
evaluates cluster quality using silhouette scores, and visualizes the results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import argparse
from kmeans import KMeans


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Cluster Analysis for Diabetic Patient Data')
    parser.add_argument('--data', type=str, default='data/diabetic_data.csv',
                        help='Path to the diabetic data CSV file')
    parser.add_argument('--min-k', type=int, default=2,
                        help='Minimum number of clusters to test')
    parser.add_argument('--max-k', type=int, default=6,
                        help='Maximum number of clusters to test')
    parser.add_argument('--pca-components', type=int, default=50,
                        help='Number of PCA components for clustering')
    parser.add_argument('--vis-dims', type=int, default=2, choices=[2, 3],
                        help='Number of dimensions for visualization (2 or 3)')
    parser.add_argument('--random-seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    return args


def load_and_preprocess_data(data_path):
    """
    Load and preprocess the diabetic patient data.
    This function mimics the preprocessing steps from EDA_Preprocessing.ipynb
    """
    print("Loading data...")
    df = pd.read_csv(data_path)
    print(f"Original dataset shape: {df.shape}")
    
    # Replace '?' placeholders with NaN
    df = df.replace('?', np.nan)
    
    # Drop columns with >40% missing values
    threshold = 0.4 * len(df)
    df = df.dropna(thresh=threshold, axis=1)
    
    # Fill remaining missing values for categorical features with 'Unknown'
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna('Unknown')
    
    # Map target variable 'readmitted' (we'll preserve it for analysis)
    df['readmitted'] = df['readmitted'].map({'NO': 0, '>30': 1, '<30': 2})
    
    # Encode categorical variables
    cat_cols = df.select_dtypes(include='object').columns
    le = LabelEncoder()
    
    for col in cat_cols:
        if df[col].nunique() < 10:
            df[col] = le.fit_transform(df[col].astype(str))
        else:
            df = pd.get_dummies(df, columns=[col], drop_first=True)
    
    print(f"After encoding, dataset shape: {df.shape}")
    
    # Separate features from target (if we want to exclude target from clustering)
    # For unsupervised clustering, we can exclude the target variable
    if 'readmitted' in df.columns:
        target = df['readmitted'].copy()
        features_df = df.drop(columns=['readmitted'])
    else:
        features_df = df
        target = None
    
    # Also drop ID columns if they exist (they shouldn't be used for clustering)
    id_cols = ['encounter_id', 'patient_nbr']
    for col in id_cols:
        if col in features_df.columns:
            features_df = features_df.drop(columns=[col])
    
    # Normalize numeric features
    num_cols = features_df.select_dtypes(include=['int64', 'float64']).columns
    scaler = StandardScaler()
    features_df[num_cols] = scaler.fit_transform(features_df[num_cols])
    
    # Convert to numpy array
    X = features_df.values
    
    print(f"Final feature matrix shape: {X.shape}")
    print("Preprocessing complete!\n")
    
    return X, target, features_df


def apply_pca(X, n_components):
    """Apply PCA for dimensionality reduction."""
    print(f"Applying PCA with {n_components} components...")
    pca_model = PCA(n_components=n_components)
    X_pca = pca_model.fit_transform(X)
    explained_variance = np.sum(pca_model.explained_variance_ratio_)
    print(f"PCA complete. Explained variance: {explained_variance:.4f} ({explained_variance*100:.2f}%)\n")
    return X_pca, pca_model


def visualize_cluster(X_vis, clustering, title, save_path, silhouette_score=None):
    """
    Visualizes clusters in 2D or 3D scatter plot with color-coded clusters.
    :param X_vis: Data points for visualization (n_samples, 2 or 3)
    :param clustering: cluster labels for each data point
    :param title: Title for the plot
    :param save_path: Path to save the plot
    :param silhouette_score: Optional silhouette score to display
    """
    n_dims = X_vis.shape[1]
    
    # Get unique cluster labels and sort them
    unique_clusters = np.unique(clustering)
    n_clusters = len(unique_clusters)
    
    # Use a colormap to assign colors to clusters
    if hasattr(matplotlib, 'colormaps'):
        cmap = matplotlib.colormaps['tab20']
    else:
        cmap = plt.cm.get_cmap('tab20')
    
    if n_dims == 2:
        # 2D visualization
        plt.figure(figsize=(10, 8))
        
        for i, cluster_id in enumerate(unique_clusters):
            mask = (clustering == cluster_id)
            color_value = i / max(n_clusters - 1, 1) if n_clusters > 1 else 0
            color = cmap(color_value)
            plt.scatter(X_vis[mask, 0], X_vis[mask, 1], color=color, 
                       label=f'Cluster {int(cluster_id)}', 
                       alpha=0.6, s=20, edgecolors='black', linewidths=0.5)
        
        plt.xlabel('Principal Component 1', fontsize=12)
        plt.ylabel('Principal Component 2', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        
    elif n_dims == 3:
        # 3D visualization
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        for i, cluster_id in enumerate(unique_clusters):
            mask = (clustering == cluster_id)
            color_value = i / max(n_clusters - 1, 1) if n_clusters > 1 else 0
            color = cmap(color_value)
            ax.scatter(X_vis[mask, 0], X_vis[mask, 1], X_vis[mask, 2],
                      color=color, label=f'Cluster {int(cluster_id)}',
                      alpha=0.6, s=20, edgecolors='black', linewidths=0.5)
        
        ax.set_xlabel('Principal Component 1', fontsize=11)
        ax.set_ylabel('Principal Component 2', fontsize=11)
        ax.set_zlabel('Principal Component 3', fontsize=11)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    
    # Add silhouette score to title if provided
    if silhouette_score is not None:
        title = f'{title}\nSilhouette Score: {silhouette_score:.4f}'
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main function to perform cluster analysis."""
    args = parse_args()
    
    # Set random seed for reproducibility
    np.random.seed(args.random_seed)
    
    # Load and preprocess data
    X, target, features_df = load_and_preprocess_data(args.data)
    
    # Apply PCA for dimensionality reduction (clustering on reduced space)
    X_pca, pca_model = apply_pca(X, args.pca_components)
    
    print("=" * 70)
    print("CLUSTER ANALYSIS: Testing K-Means (KMeans++) with different k values")
    print("=" * 70)
    
    # Test different k values
    ks = list(range(args.min_k, args.max_k + 1))
    silhouettes = []
    
    best_k = None
    best_score = -1
    best_clustering = None
    
    for k in ks:
        print(f"\nTesting k={k}...", end=' ', flush=True)
        
        # KMeans++ initialization
        kmeans = KMeans(n_clusters=k, init='kmeans++')
        clustering = kmeans.fit(X_pca)
        sil_score = kmeans.silhouette(clustering, X_pca)
        silhouettes.append(sil_score)
        print(f"Silhouette: {sil_score:.4f}")
        
        if sil_score > best_score:
            best_score = sil_score
            best_k = k
            best_clustering = clustering.copy()
    
    # Print results summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    print("\nKMeans++ Results")
    print("-" * 70)
    print("k\tSilhouette Coefficient")
    print("-" * 70)
    for k, score in zip(ks, silhouettes):
        marker = " <-- Best" if k == best_k else ""
        print(f"{k}\t{score:.4f}{marker}")
    print(f"\nBest k: {best_k} (Silhouette: {best_score:.4f})")
    
    # Plot silhouette coefficients
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)
    
    print("\nGenerating silhouette score plot...")
    plt.figure(figsize=(10, 6))
    plt.plot(ks, silhouettes, 'o-', label='KMeans++', linewidth=2, markersize=8, color='steelblue')
    plt.xlabel('Number of Clusters (k)', fontsize=12)
    plt.ylabel('Silhouette Coefficient', fontsize=12)
    plt.title('Silhouette Coefficient vs Number of Clusters (KMeans++)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(ks)
    plt.tight_layout()
    plt.savefig('silhouette_scores.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved silhouette score plot to silhouette_scores.png")
    
    # Reduce to 2D or 3D for visualization only (clustering was done with PCA dimensions)
    print(f"\nReducing to {args.vis_dims}D for cluster visualization...")
    X_vis = PCA(n_components=args.vis_dims).fit_transform(X)
    
    # Visualize best clustering
    print(f"\nVisualizing clusters with k={best_k}...")
    title = f'K-Means Clustering (KMeans++, k={best_k})'
    visualize_cluster(X_vis, best_clustering, 
                     title, f'clustering_k{best_k}.png',
                     silhouette_score=best_score)
    print(f"  Saved plot to clustering_k{best_k}.png")
    
    # If target variable exists, create comparison plot (2D only)
    if target is not None and args.vis_dims == 2:
        print("\nCreating comparison plot with actual readmission labels...")
        plt.figure(figsize=(10, 8))
        
        # Plot with actual readmission labels
        unique_targets = np.unique(target)
        colors_target = ['red', 'orange', 'green']
        labels_target = ['No Readmission', '>30 days', '<30 days']
        
        for i, target_val in enumerate(unique_targets):
            mask = (target == target_val)
            plt.scatter(X_vis[mask, 0], X_vis[mask, 1], 
                       color=colors_target[i], label=labels_target[i],
                       alpha=0.6, s=20, edgecolors='black', linewidths=0.5)
        
        plt.xlabel('Principal Component 1', fontsize=12)
        plt.ylabel('Principal Component 2', fontsize=12)
        plt.title('PCA Projection Colored by Readmission Status', fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.tight_layout()
        plt.savefig('readmission_labels_pca.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  Saved readmission labels comparison plot to readmission_labels_pca.png")
    
    print("\n" + "=" * 70)
    print("CLUSTER ANALYSIS COMPLETE!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - silhouette_scores.png")
    print(f"  - clustering_k{best_k}.png")
    if target is not None and args.vis_dims == 2:
        print("  - readmission_labels_pca.png")
    print()


if __name__ == '__main__':
    main()

