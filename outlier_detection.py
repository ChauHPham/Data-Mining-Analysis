"""
Outlier Detection Analysis
Implements two methods: Isolation Forest and Local Outlier Factor (LOF)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
import argparse
import os


def load_and_preprocess_data(file_path):
    """
    Load and preprocess the diabetic data
    Returns: X (features), y (target), feature_names
    """
    print("Loading data...")
    df = pd.read_csv(file_path)
    print(f"Initial shape: {df.shape}")
    
    # Drop duplicates
    df = df.drop_duplicates()
    print(f"After removing duplicates: {df.shape}")
    
    # Handle '?' as missing values
    df = df.replace('?', np.nan)
    
    # Drop columns with >50% missing
    threshold = 0.5
    missing_pct = df.isnull().sum() / len(df)
    cols_to_drop = missing_pct[missing_pct > threshold].index.tolist()
    df = df.drop(columns=cols_to_drop)
    print(f"Dropped {len(cols_to_drop)} columns with >50% missing")
    
    # Drop specific columns
    drop_cols = ['encounter_id', 'patient_nbr']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])
    
    # Target variable
    if 'readmitted' in df.columns:
        y = df['readmitted'].copy()
        df = df.drop(columns=['readmitted'])
    else:
        y = None
    
    # Identify categorical and numerical columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    print(f"Categorical columns: {len(categorical_cols)}")
    print(f"Numerical columns: {len(numerical_cols)}")
    
    # Handle categorical columns
    for col in categorical_cols:
        if df[col].nunique() > 10:
            # One-hot encode high cardinality
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df, dummies], axis=1)
            df = df.drop(columns=[col])
        else:
            # Label encode low cardinality
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    
    # Fill missing numerical values with median
    for col in numerical_cols:
        if col in df.columns and df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)
    
    # Fill any remaining missing values
    df = df.fillna(0)
    
    X = df.copy()
    feature_names = X.columns.tolist()
    
    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=feature_names)
    
    print(f"Final shape: {X.shape}")
    print(f"Total features: {len(feature_names)}")
    
    return X, y, feature_names


def apply_isolation_forest(X, contamination=0.01, random_state=42, max_samples='auto'):
    """
    Apply Isolation Forest for outlier detection
    """
    print("\n" + "="*70)
    print("ISOLATION FOREST")
    print("="*70)
    
    iso = IsolationForest(
        n_estimators=100, 
        contamination=contamination, 
        max_samples=max_samples,  # 'auto' uses min(256, n_samples)
        random_state=random_state,
        n_jobs=-1
    )
    
    print(f"Fitting Isolation Forest (contamination={contamination}, max_samples={max_samples})...")
    X_array = X.values if isinstance(X, pd.DataFrame) else X
    iso.fit(X_array)
    
    # Predict: -1 for outliers, 1 for inliers
    outlier_preds = iso.predict(X_array)
    outlier_indices = np.where(outlier_preds == -1)[0]
    
    # Decision function: lower scores = more likely outlier
    scores = iso.decision_function(X_array)
    
    print(f"✓ Detected {len(outlier_indices)} outliers out of {len(X_array)} samples ({len(outlier_indices)/len(X_array)*100:.2f}%)")
    print(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")
    print(f"  Top 5 outlier scores (lowest): {np.partition(scores, min(5, len(scores)))[:5]}")
    
    return outlier_preds, scores, outlier_indices


def apply_lof(X, n_neighbors=20, contamination=0.01, max_samples=10000, random_state=42):
    """
    Apply Local Outlier Factor for outlier detection
    Note: LOF can be slow on large datasets, so we optionally sample
    """
    print("\n" + "="*70)
    print("LOCAL OUTLIER FACTOR (LOF)")
    print("="*70)
    
    # Sample if dataset is too large
    X_array = X.values if isinstance(X, pd.DataFrame) else X
    if len(X_array) > max_samples:
        print(f"⚠ Dataset has {len(X_array)} samples, sampling {max_samples} for LOF (for speed)...")
        np.random.seed(random_state)
        indices = np.random.choice(len(X_array), max_samples, replace=False)
        X_sampled = X_array[indices]
        print(f"  Using {len(X_sampled)} samples")
    else:
        X_sampled = X_array
        indices = np.arange(len(X_array))
    
    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
        n_jobs=-1
    )
    
    print(f"Fitting LOF (n_neighbors={n_neighbors}, contamination={contamination})...")
    # LOF fit_predict does both fitting and prediction
    outlier_preds_sampled = lof.fit_predict(X_sampled)
    outlier_indices_sampled = np.where(outlier_preds_sampled == -1)[0]
    
    # Negative outlier factor: more negative = more outlier-like
    scores_sampled = lof.negative_outlier_factor_
    
    # Map back to original indices
    if len(X_array) > max_samples:
        # Create full array with default inlier prediction
        outlier_preds = np.ones(len(X_array), dtype=int)
        outlier_preds[indices] = outlier_preds_sampled
        
        scores = np.ones(len(X_array))
        scores[indices] = scores_sampled
        
        outlier_indices = indices[outlier_indices_sampled]
    else:
        outlier_preds = outlier_preds_sampled
        scores = scores_sampled
        outlier_indices = outlier_indices_sampled
    
    print(f"✓ Detected {len(outlier_indices)} outliers out of {len(X_array)} samples ({len(outlier_indices)/len(X_array)*100:.2f}%)")
    print(f"  Score range: [{scores[scores != 1].min():.4f}, {scores[scores != 1].max():.4f}]")
    outlier_scores = scores[outlier_preds == -1]
    if len(outlier_scores) >= 5:
        print(f"  Top 5 outlier scores (most negative): {np.partition(outlier_scores, min(5, len(outlier_scores)))[:5]}")
    print(f"\nLOF Interpretation:")
    print(f"  - LOF measures local density deviation")
    print(f"  - Points in sparse regions (low local density) are flagged as outliers")
    print(f"  - More negative scores = stronger outlier")
    
    return outlier_preds, scores, outlier_indices


def visualize_outliers(X, methods_results, output_dir='outlier_plots'):
    """
    Visualize outliers detected by each method using PCA
    """
    print("\n" + "="*70)
    print("VISUALIZATION")
    print("="*70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Apply PCA for 2D visualization
    print("Applying PCA for 2D visualization...")
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)
    print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")
    
    # Create subplots for all methods
    n_methods = len(methods_results)
    fig, axes = plt.subplots(1, n_methods + 1, figsize=(6*(n_methods+1), 5))
    
    # Plot 1: All data
    axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c='blue', alpha=0.3, s=20, label='All points')
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    axes[0].set_title('All Data Points')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot each method
    for idx, (method_name, results) in enumerate(methods_results.items(), start=1):
        outlier_preds, scores, outlier_indices = results
        
        # Inliers vs Outliers
        inlier_mask = outlier_preds == 1
        outlier_mask = outlier_preds == -1
        
        axes[idx].scatter(X_pca[inlier_mask, 0], X_pca[inlier_mask, 1], 
                         c='blue', alpha=0.3, s=20, label='Inliers')
        axes[idx].scatter(X_pca[outlier_mask, 0], X_pca[outlier_mask, 1], 
                         c='red', alpha=0.8, s=50, marker='x', label='Outliers')
        axes[idx].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        axes[idx].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        axes[idx].set_title(f'{method_name}\n({len(outlier_indices)} outliers)')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'outliers_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved comparison plot: {plot_path}")
    plt.close()
    
    # Individual detailed plots for each method
    for method_name, results in methods_results.items():
        outlier_preds, scores, outlier_indices = results
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Outliers highlighted
        inlier_mask = outlier_preds == 1
        outlier_mask = outlier_preds == -1
        
        axes[0].scatter(X_pca[inlier_mask, 0], X_pca[inlier_mask, 1], 
                       c='blue', alpha=0.3, s=20, label='Inliers')
        axes[0].scatter(X_pca[outlier_mask, 0], X_pca[outlier_mask, 1], 
                       c='red', alpha=0.8, s=50, marker='x', label='Outliers')
        axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        axes[0].set_title(f'{method_name}: Outliers vs Inliers')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Score distribution
        axes[1].hist(scores, bins=50, edgecolor='black', alpha=0.7)
        axes[1].axvline(scores[outlier_mask].max() if outlier_mask.any() else scores.min(), 
                       color='red', linestyle='--', linewidth=2, label='Outlier threshold')
        axes[1].set_xlabel('Anomaly Score')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title(f'{method_name}: Score Distribution')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f'outliers_{method_name.lower().replace(" ", "_")}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved {method_name} plot: {plot_path}")
        plt.close()


def analyze_overlap(methods_results):
    """
    Analyze overlap between outliers detected by different methods
    """
    print("\n" + "="*70)
    print("OUTLIER OVERLAP ANALYSIS")
    print("="*70)
    
    method_names = list(methods_results.keys())
    outlier_sets = {name: set(results[2]) for name, results in methods_results.items()}
    
    # Pairwise overlap
    print("\nPairwise Overlap:")
    for i, name1 in enumerate(method_names):
        for name2 in method_names[i+1:]:
            overlap = outlier_sets[name1] & outlier_sets[name2]
            union = outlier_sets[name1] | outlier_sets[name2]
            jaccard = len(overlap) / len(union) if len(union) > 0 else 0
            print(f"  {name1} ∩ {name2}: {len(overlap)} outliers (Jaccard: {jaccard:.3f})")
    
    # Common outliers (detected by all methods)
    common_outliers = set.intersection(*outlier_sets.values())
    print(f"\n✓ Outliers detected by ALL methods: {len(common_outliers)}")
    
    # Unique outliers (detected by only one method)
    print("\nUnique outliers (detected by only one method):")
    for name in method_names:
        others = set.union(*[outlier_sets[n] for n in method_names if n != name])
        unique = outlier_sets[name] - others
        print(f"  {name} only: {len(unique)} outliers")
    
    return common_outliers


def main():
    parser = argparse.ArgumentParser(description='Outlier Detection Analysis')
    parser.add_argument('--data', type=str, default='data/diabetic_data.csv',
                        help='Path to the dataset')
    parser.add_argument('--contamination', type=float, default=0.01,
                        help='Expected proportion of outliers (default: 0.01 = 1%%)')
    parser.add_argument('--lof-neighbors', type=int, default=20,
                        help='Number of neighbors for LOF (default: 20)')
    parser.add_argument('--lof-max-samples', type=int, default=10000,
                        help='Max samples for LOF (for speed on large datasets, default: 10000)')
    parser.add_argument('--output-dir', type=str, default='outlier_plots',
                        help='Directory to save plots')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random state for reproducibility')
    
    args = parser.parse_args()
    
    print("="*70)
    print("OUTLIER DETECTION ANALYSIS")
    print("="*70)
    print(f"Dataset: {args.data}")
    print(f"Contamination: {args.contamination} ({args.contamination*100:.1f}%)")
    print(f"LOF neighbors: {args.lof_neighbors}")
    print(f"Random state: {args.random_state}")
    print("="*70)
    
    # Load and preprocess data
    X, y, feature_names = load_and_preprocess_data(args.data)
    
    # Apply both methods
    methods_results = {}
    
    # 1. Isolation Forest
    iso_preds, iso_scores, iso_indices = apply_isolation_forest(
        X, contamination=args.contamination, random_state=args.random_state
    )
    methods_results['Isolation Forest'] = (iso_preds, iso_scores, iso_indices)
    
    # 2. Local Outlier Factor
    lof_preds, lof_scores, lof_indices = apply_lof(
        X, n_neighbors=args.lof_neighbors, contamination=args.contamination,
        max_samples=args.lof_max_samples, random_state=args.random_state
    )
    methods_results['LOF'] = (lof_preds, lof_scores, lof_indices)
    
    # Visualize outliers
    visualize_outliers(X, methods_results, output_dir=args.output_dir)
    
    # Analyze overlap
    common_outliers = analyze_overlap(methods_results)
    
    # Final summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total samples: {len(X)}")
    for method_name, results in methods_results.items():
        n_outliers = len(results[2])
        print(f"{method_name}: {n_outliers} outliers ({n_outliers/len(X)*100:.2f}%)")
    if len(methods_results) > 1:
        print(f"\nCommon outliers (both methods): {len(common_outliers)}")
    print("\nRecommendation:")
    print("  - Review common outliers first (high confidence)")
    print("  - Investigate if outliers are:")
    print("    • Data errors/noise → Remove")
    print("    • Valid rare events → Keep or handle specially")
    print(f"\nPlots saved to: {args.output_dir}/")
    print("="*70)


if __name__ == '__main__':
    main()

