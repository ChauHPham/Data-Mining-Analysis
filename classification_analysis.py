"""
Classification Analysis for Diabetic Patient Data
CMPT 459 Course Project

This script performs classification on the diabetic patient dataset using Random Forest.
It includes train/test split, cross-validation, and evaluation metrics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
import argparse
from random_forest import RandomForest


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Classification Analysis for Diabetic Patient Data')
    parser.add_argument('--data', type=str, default='data/diabetic_data.csv',
                        help='Path to the diabetic data CSV file')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Proportion of data to use for testing (default: 0.2)')
    parser.add_argument('--n-estimators', type=int, default=20,
                        help='Number of trees in the Random Forest (default: 20)')
    parser.add_argument('--max-depth', type=int, default=10,
                        help='Maximum depth of trees (default: 10)')
    parser.add_argument('--min-samples-split', type=int, default=2,
                        help='Minimum samples required to split a node (default: 2)')
    parser.add_argument('--criterion', type=str, default='gini', choices=['gini', 'entropy'],
                        help='Splitting criterion (default: gini)')
    parser.add_argument('--max-features', type=str, default='sqrt',
                        help='Number of features to consider at each split: sqrt, log2, or int (default: sqrt)')
    parser.add_argument('--cv-folds', type=int, default=5, choices=[5, 10],
                        help='Number of folds for cross-validation (default: 5)')
    parser.add_argument('--n-features', type=int, default=100,
                        help='Number of best features to select (default: 100)')
    parser.add_argument('--random-seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    return args


def load_and_preprocess_data(data_path):
    """
    Load and preprocess the diabetic patient data for classification.
    This function adapts the preprocessing from cluster_analysis.py for classification tasks.
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
    
    # Prepare target variable 'readmitted' for classification
    # Map to binary classification: NO = 0, readmitted (>30 or <30) = 1
    # Or we can keep it as multi-class: NO=0, >30=1, <30=2
    # Let's use binary classification for simplicity
    df['readmitted_binary'] = df['readmitted'].map({'NO': 0, '>30': 1, '<30': 1})
    
    # Also keep original multi-class target for comparison
    df['readmitted_multi'] = df['readmitted'].map({'NO': 0, '>30': 1, '<30': 2})
    
    # Encode categorical variables
    cat_cols = df.select_dtypes(include='object').columns
    
    # Remove target columns from categorical columns to encode
    cat_cols = cat_cols.drop(['readmitted'], errors='ignore')
    
    le = LabelEncoder()
    encoded_df = df.copy()
    
    for col in cat_cols:
        if encoded_df[col].nunique() < 10:
            # Label encode for columns with few unique values
            encoded_df[col] = le.fit_transform(encoded_df[col].astype(str))
        else:
            # One-hot encode for columns with many unique values
            encoded_df = pd.get_dummies(encoded_df, columns=[col], drop_first=True, prefix=col[:10])
    
    print(f"After encoding, dataset shape: {encoded_df.shape}")
    
    # Separate features from target
    # Drop ID columns if they exist
    id_cols = ['encounter_id', 'patient_nbr']
    target_cols = ['readmitted', 'readmitted_binary', 'readmitted_multi']
    
    # Get feature columns (exclude IDs and targets)
    feature_cols = [col for col in encoded_df.columns 
                    if col not in id_cols + target_cols]
    
    X = encoded_df[feature_cols]
    y_binary = encoded_df['readmitted_binary']
    y_multi = encoded_df['readmitted_multi']
    
    # Normalize numeric features
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns
    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])
    
    print(f"Final feature matrix shape: {X.shape}")
    print(f"Target distribution (binary):")
    print(y_binary.value_counts().sort_index())
    print(f"\nTarget distribution (multi-class):")
    print(y_multi.value_counts().sort_index())
    print("Preprocessing complete!\n")
    
    return X, y_binary, y_multi


def evaluate_classifier(y_true, y_pred, target_names=None):
    """
    Calculate and print classification metrics.
    
    :param y_true: True labels
    :param y_pred: Predicted labels
    :param target_names: Names of target classes (for classification report)
    :return: Dictionary of metrics
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    
    # Print classification report
    if target_names is not None:
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))
    
    return metrics


def plot_confusion_matrix(y_true, y_pred, title, save_path, target_names=None):
    """
    Plot and save confusion matrix.
    
    :param y_true: True labels
    :param y_pred: Predicted labels
    :param title: Plot title
    :param save_path: Path to save the plot
    :param target_names: Names of target classes
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names if target_names else None,
                yticklabels=target_names if target_names else None,
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main function to perform classification analysis."""
    args = parse_args()
    
    # Set random seed for reproducibility
    np.random.seed(args.random_seed)
    
    # Load and preprocess data
    X, y_binary, y_multi = load_and_preprocess_data(args.data)
    
    print("=" * 70)
    print("CLASSIFICATION ANALYSIS: Random Forest")
    print("=" * 70)
    
    # Use binary classification (can be changed to multi-class)
    y = y_binary
    target_names = ['No Readmission', 'Readmission']
    
    print(f"\nUsing binary classification target (Readmission: No=0, Yes=1)")
    print(f"Class distribution:")
    print(y.value_counts().sort_index())
    
    # Feature selection to reduce dimensionality (SPEED UP!)
    print("\n" + "=" * 70)
    print(f"FEATURE SELECTION")
    print("=" * 70)
    original_features = X.shape[1]
    n_features_to_select = min(args.n_features, original_features)
    
    print(f"Original features: {original_features}")
    print(f"Selecting top {n_features_to_select} features using ANOVA F-test...")
    
    selector = SelectKBest(f_classif, k=n_features_to_select)
    X_selected = selector.fit_transform(X, y)
    
    print(f"Reduced to: {X_selected.shape[1]} features")
    print(f"Speed improvement: ~{original_features / n_features_to_select:.1f}x faster!\n")
    
    # Convert back to DataFrame for consistency
    selected_feature_indices = selector.get_support(indices=True)
    selected_feature_names = X.columns[selected_feature_indices]
    X = pd.DataFrame(X_selected, columns=selected_feature_names, index=X.index)
    
    # Train/Test Split
    print("=" * 70)
    print(f"TRAIN/TEST SPLIT ({1-args.test_size:.0%} train, {args.test_size:.0%} test)")
    print("=" * 70)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_seed, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Testing set size:  {X_test.shape[0]} samples, {X_test.shape[1]} features")
    print(f"\nTraining set class distribution:")
    print(y_train.value_counts().sort_index())
    print(f"\nTesting set class distribution:")
    print(y_test.value_counts().sort_index())
    
    # Initialize Random Forest
    print("\n" + "=" * 70)
    print("RANDOM FOREST CONFIGURATION")
    print("=" * 70)
    print(f"  Number of trees:      {args.n_estimators}")
    print(f"  Max depth:            {args.max_depth if args.max_depth else 'Unlimited'}")
    print(f"  Min samples split:    {args.min_samples_split}")
    print(f"  Criterion:            {args.criterion}")
    print(f"  Max features:         {args.max_features}")
    print(f"  Random seed:          {args.random_seed}")
    
    # Convert max_features to appropriate type
    if args.max_features.isdigit():
        max_features = int(args.max_features)
    elif args.max_features.lower() in ['sqrt', 'log2']:
        max_features = args.max_features.lower()
    else:
        max_features = args.max_features
    
    # Train Random Forest
    rf = RandomForest(
        n_estimators=args.n_estimators,
        criterion=args.criterion,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        max_features=max_features,
        random_state=args.random_seed
    )
    
    print("\n" + "=" * 70)
    print("TRAINING RANDOM FOREST")
    print("=" * 70)
    rf.fit(X_train, y_train)
    
    # Evaluate on training set
    print("\n" + "=" * 70)
    print("TRAINING SET EVALUATION")
    print("=" * 70)
    y_train_pred = rf.predict(X_train)
    train_metrics = evaluate_classifier(y_train, y_train_pred, target_names)
    
    # Evaluate on test set
    print("\n" + "=" * 70)
    print("TEST SET EVALUATION")
    print("=" * 70)
    y_test_pred = rf.predict(X_test)
    test_metrics = evaluate_classifier(y_test, y_test_pred, target_names)
    
    # Cross-Validation
    print("\n" + "=" * 70)
    print(f"CROSS-VALIDATION ({args.cv_folds}-fold)")
    print("=" * 70)
    
    # Create a new Random Forest for CV (to avoid using the already trained one)
    # We'll use the same hyperparameters
    kfold = KFold(n_splits=args.cv_folds, shuffle=True, random_state=args.random_seed)
    
    cv_scores = []
    print(f"\nPerforming {args.cv_folds}-fold cross-validation...")
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train), 1):
        print(f"  Fold {fold}/{args.cv_folds}...", end=' ', flush=True)
        
        X_cv_train = X_train.iloc[train_idx]
        y_cv_train = y_train.iloc[train_idx]
        X_cv_val = X_train.iloc[val_idx]
        y_cv_val = y_train.iloc[val_idx]
        
        # Train a new RF for this fold
        rf_cv = RandomForest(
            n_estimators=args.n_estimators,
            criterion=args.criterion,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            max_features=max_features,
            random_state=args.random_seed + fold  # Different seed for each fold
        )
        rf_cv.fit(X_cv_train, y_cv_train)
        
        # Evaluate on validation set
        y_cv_pred = rf_cv.predict(X_cv_val)
        cv_accuracy = accuracy_score(y_cv_val, y_cv_pred)
        cv_scores.append(cv_accuracy)
        print(f"Accuracy: {cv_accuracy:.4f}")
    
    print("\nCross-Validation Results:")
    print("-" * 70)
    print("Fold\tAccuracy")
    print("-" * 70)
    for i, score in enumerate(cv_scores, 1):
        print(f"{i}\t{score:.4f}")
    
    mean_cv_score = np.mean(cv_scores)
    std_cv_score = np.std(cv_scores)
    print(f"\nMean CV Accuracy: {mean_cv_score:.4f} (+/- {std_cv_score:.4f})")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nTraining Accuracy:    {train_metrics['accuracy']:.4f}")
    print(f"Testing Accuracy:     {test_metrics['accuracy']:.4f}")
    print(f"CV Mean Accuracy:     {mean_cv_score:.4f} (+/- {std_cv_score:.4f})")
    print(f"\nTesting F1 Score:     {test_metrics['f1_score']:.4f}")
    print(f"Testing Precision:    {test_metrics['precision']:.4f}")
    print(f"Testing Recall:        {test_metrics['recall']:.4f}")
    
    # Visualizations
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)
    
    # Confusion matrix for test set
    print("\nGenerating confusion matrix for test set...")
    plot_confusion_matrix(
        y_test, y_test_pred,
        title=f'Confusion Matrix - Test Set (Accuracy: {test_metrics["accuracy"]:.4f})',
        save_path='confusion_matrix_test.png',
        target_names=target_names
    )
    print("  Saved confusion matrix to confusion_matrix_test.png")
    
    # Confusion matrix for training set
    print("\nGenerating confusion matrix for training set...")
    plot_confusion_matrix(
        y_train, y_train_pred,
        title=f'Confusion Matrix - Training Set (Accuracy: {train_metrics["accuracy"]:.4f})',
        save_path='confusion_matrix_train.png',
        target_names=target_names
    )
    print("  Saved confusion matrix to confusion_matrix_train.png")
    
    # Cross-validation scores plot
    print("\nGenerating cross-validation scores plot...")
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, args.cv_folds + 1), cv_scores, color='steelblue', alpha=0.7, edgecolor='black')
    plt.axhline(y=mean_cv_score, color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {mean_cv_score:.4f}')
    plt.xlabel('Fold', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title(f'{args.cv_folds}-Fold Cross-Validation Scores', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(range(1, args.cv_folds + 1))
    plt.ylim([0, 1.0])
    plt.tight_layout()
    plt.savefig('cv_scores.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved cross-validation scores plot to cv_scores.png")
    
    print("\n" + "=" * 70)
    print("CLASSIFICATION ANALYSIS COMPLETE!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - confusion_matrix_test.png")
    print("  - confusion_matrix_train.png")
    print("  - cv_scores.png")
    print()


if __name__ == '__main__':
    main()

