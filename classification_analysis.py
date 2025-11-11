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
from sklearn.model_selection import train_test_split, cross_val_score, KFold, RandomizedSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                              confusion_matrix, classification_report, roc_curve, 
                              roc_auc_score, auc)
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
import argparse
import scipy.stats as st
import time
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
    parser.add_argument('--random-seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--tune-hyperparameters', action='store_true',
                        help='Enable hyperparameter tuning using RandomizedSearchCV')
    parser.add_argument('--tuning-iterations', type=int, default=20,
                        help='Number of iterations for RandomizedSearchCV (default: 20)')
    parser.add_argument('--feature-selection', type=str, default='lasso', 
                        choices=['lasso', 'none'],
                        help='Feature selection method: lasso (L1) or none (default: lasso)')
    parser.add_argument('--lasso-C', type=float, default=0.1,
                        help='Inverse regularization strength for Lasso (default: 0.1)')
    
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


def evaluate_classifier(y_true, y_pred, y_proba=None, target_names=None):
    """
    Calculate and print classification metrics.
    
    :param y_true: True labels
    :param y_pred: Predicted labels
    :param y_proba: Predicted probabilities (optional, for AUC-ROC)
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
    
    # Calculate AUC-ROC if probabilities are provided
    if y_proba is not None:
        try:
            # For binary classification
            if y_proba.shape[1] == 2:
                auc_roc = roc_auc_score(y_true, y_proba[:, 1])
                print(f"  AUC-ROC:   {auc_roc:.4f}")
                metrics['auc_roc'] = auc_roc
            # For multi-class classification (one-vs-rest)
            else:
                auc_roc = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
                print(f"  AUC-ROC:   {auc_roc:.4f} (weighted one-vs-rest)")
                metrics['auc_roc'] = auc_roc
        except Exception as e:
            print(f"  AUC-ROC:   Could not calculate ({str(e)})")
            metrics['auc_roc'] = None
    
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


def plot_roc_curve(y_true, y_proba, title, save_path, target_names=None):
    """
    Plot and save ROC curve(s).
    
    :param y_true: True labels
    :param y_proba: Predicted probabilities
    :param title: Plot title
    :param save_path: Path to save the plot
    :param target_names: Names of target classes
    """
    n_classes = y_proba.shape[1]
    
    plt.figure(figsize=(10, 8))
    
    if n_classes == 2:
        # Binary classification - single ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier (AUC = 0.5000)')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=11)
        plt.grid(True, alpha=0.3)
        
    else:
        # Multi-class classification - one ROC curve per class
        from sklearn.preprocessing import label_binarize
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        
        # Compute ROC curve and AUC for each class
        colors = plt.cm.get_cmap('tab10')(range(n_classes))
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            class_name = target_names[i] if target_names else f'Class {i}'
            plt.plot(fpr, tpr, color=colors[i], lw=2,
                    label=f'{class_name} (AUC = {roc_auc:.4f})')
        
        # Compute micro-average ROC curve and AUC
        fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_proba.ravel())
        roc_auc_micro = auc(fpr_micro, tpr_micro)
        plt.plot(fpr_micro, tpr_micro, color='deeppink', lw=2, linestyle=':',
                label=f'Micro-average (AUC = {roc_auc_micro:.4f})')
        
        # Plot random classifier line
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                label='Random Classifier (AUC = 0.5000)')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=9)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def select_features_lasso(X, y, C=0.1, n_features_target=None, random_state=42):
    """
    Select features using Lasso (L1) regularization.
    
    :param X: Feature matrix
    :param y: Target labels
    :param C: Inverse regularization strength (smaller values = stronger regularization)
    :param n_features_target: Target number of features (will try to adjust C to get close)
    :param random_state: Random seed
    :return: Selected feature indices, feature names, selector, number of features selected
    """
    print("\n" + "=" * 70)
    print("LASSO (L1) FEATURE SELECTION")
    print("=" * 70)
    
    print(f"\nTraining Logistic Regression with L1 penalty...")
    print(f"  Regularization strength C: {C}")
    print(f"  Solver: liblinear")
    
    # Train logistic regression with L1 penalty
    logreg_l1 = LogisticRegression(
        penalty='l1',
        C=C,
        solver='liblinear',
        random_state=random_state,
        max_iter=1000
    )
    
    logreg_l1.fit(X, y)
    
    # Get feature importances (absolute coefficients)
    if len(logreg_l1.classes_) == 2:
        # Binary classification
        coefficients = np.abs(logreg_l1.coef_[0])
    else:
        # Multi-class: average absolute coefficients across classes
        coefficients = np.mean(np.abs(logreg_l1.coef_), axis=0)
    
    # Use SelectFromModel to select features with non-zero coefficients
    selector = SelectFromModel(logreg_l1, prefit=True, threshold=1e-5)
    selected_mask = selector.get_support()
    selected_indices = np.where(selected_mask)[0]
    selected_feature_names = X.columns[selected_indices].tolist()
    
    n_selected = len(selected_indices)
    
    print(f"\n✓ Lasso selected {n_selected} features out of {X.shape[1]} ({n_selected/X.shape[1]*100:.1f}%)")
    
    # Show top features by coefficient magnitude
    feature_importances = list(zip(X.columns, coefficients))
    feature_importances.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nTop 10 features by L1 coefficient magnitude:")
    print("-" * 70)
    for i, (feat, coef) in enumerate(feature_importances[:10], 1):
        selected_marker = "✓" if feat in selected_feature_names else "✗"
        print(f"  {i:2d}. {selected_marker} {feat[:50]:<50} {coef:.6f}")
    
    # Show which features were zeroed out
    zeroed_out = X.shape[1] - n_selected
    print(f"\nFeatures with zero coefficients (removed): {zeroed_out}")
    
    print(f"\nInterpretation:")
    print(f"  • L1 regularization forced {zeroed_out} feature coefficients to exactly zero")
    print(f"  • Selected features are most predictive in a linear model")
    print(f"  • Lasso tends to pick one feature from correlated groups")
    
    return selected_indices, selected_feature_names, selector, n_selected


def tune_hyperparameters(X_train, y_train, n_iter=20, cv_folds=5, random_state=42):
    """
    Perform hyperparameter tuning using RandomizedSearchCV.
    
    :param X_train: Training features
    :param y_train: Training labels
    :param n_iter: Number of parameter settings to sample
    :param cv_folds: Number of cross-validation folds
    :param random_state: Random seed
    :return: Best estimator and search results
    """
    print("\n" + "=" * 70)
    print("HYPERPARAMETER TUNING (Randomized Search)")
    print("=" * 70)
    
    # Define parameter distribution
    param_dist = {
        'n_estimators': [10, 20, 50, 100, 200],
        'max_depth': [5, 10, 15, 20, 30, None],
        'min_samples_split': [2, 5, 10, 15],
        'max_features': ['sqrt', 'log2', None],
        'criterion': ['gini', 'entropy']
    }
    
    print(f"\nParameter search space:")
    for param, values in param_dist.items():
        print(f"  {param}: {values}")
    
    print(f"\nSearch settings:")
    print(f"  Iterations: {n_iter}")
    print(f"  CV folds: {cv_folds}")
    print(f"  Scoring: f1_weighted")
    
    # Create base Random Forest
    rf_base = RandomForest(random_state=random_state)
    
    # Perform randomized search
    print(f"\nRunning RandomizedSearchCV (this may take a while)...")
    random_search = RandomizedSearchCV(
        rf_base,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv_folds,
        scoring='f1_weighted',
        random_state=random_state,
        n_jobs=-1,
        verbose=1
    )
    
    random_search.fit(X_train, y_train)
    
    print("\n" + "=" * 70)
    print("TUNING RESULTS")
    print("=" * 70)
    print(f"\nBest parameters found:")
    for param, value in random_search.best_params_.items():
        print(f"  {param}: {value}")
    
    print(f"\nBest CV F1 Score (weighted): {random_search.best_score_:.4f}")
    
    # Show top 5 parameter combinations
    results_df = pd.DataFrame(random_search.cv_results_)
    results_df = results_df.sort_values('rank_test_score')
    
    print(f"\nTop 5 parameter combinations:")
    print("-" * 70)
    for idx, (i, row) in enumerate(results_df.head(5).iterrows(), 1):
        print(f"\n{idx}. Mean F1: {row['mean_test_score']:.4f} (+/- {row['std_test_score']:.4f})")
        params = row['params']
        for param, value in params.items():
            print(f"   {param}: {value}")
    
    return random_search.best_estimator_, random_search


def compare_models(baseline_metrics, tuned_metrics, dataset_name='Test'):
    """
    Compare performance between baseline and tuned models.
    
    :param baseline_metrics: Dictionary of baseline model metrics
    :param tuned_metrics: Dictionary of tuned model metrics
    :param dataset_name: Name of dataset (e.g., 'Test', 'Training')
    """
    print("\n" + "=" * 70)
    print(f"MODEL COMPARISON ({dataset_name} Set)")
    print("=" * 70)
    
    metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
    
    print(f"\n{'Metric':<15} {'Baseline':<12} {'Tuned':<12} {'Improvement':<15}")
    print("-" * 70)
    
    for metric in metrics_to_compare:
        if metric in baseline_metrics and metric in tuned_metrics:
            baseline_val = baseline_metrics[metric]
            tuned_val = tuned_metrics[metric]
            
            if baseline_val is not None and tuned_val is not None:
                improvement = tuned_val - baseline_val
                improvement_pct = (improvement / baseline_val) * 100 if baseline_val != 0 else 0
                
                print(f"{metric.upper():<15} {baseline_val:<12.4f} {tuned_val:<12.4f} "
                      f"{improvement:+.4f} ({improvement_pct:+.2f}%)")
    
    print("\nInterpretation:")
    improvements = []
    for metric in metrics_to_compare:
        if metric in baseline_metrics and metric in tuned_metrics:
            if baseline_metrics[metric] is not None and tuned_metrics[metric] is not None:
                if tuned_metrics[metric] > baseline_metrics[metric]:
                    improvements.append(metric)
    
    if improvements:
        print(f"  ✓ Improved metrics: {', '.join(improvements)}")
        print(f"  ✓ Hyperparameter tuning successfully improved model performance")
        print(f"  ✓ The tuned model achieves better bias-variance trade-off")
    else:
        print(f"  ○ No significant improvements observed")
        print(f"  ○ Baseline hyperparameters were already near-optimal")


def plot_tuning_comparison(baseline_metrics, tuned_metrics, save_path='model_comparison.png'):
    """
    Plot comparison between baseline and tuned models.
    
    :param baseline_metrics: Dictionary of baseline model metrics
    :param tuned_metrics: Dictionary of tuned model metrics
    :param save_path: Path to save the plot
    """
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
    
    baseline_values = []
    tuned_values = []
    valid_labels = []
    
    for metric, label in zip(metrics, metric_labels):
        if metric in baseline_metrics and metric in tuned_metrics:
            if baseline_metrics[metric] is not None and tuned_metrics[metric] is not None:
                baseline_values.append(baseline_metrics[metric])
                tuned_values.append(tuned_metrics[metric])
                valid_labels.append(label)
    
    x = np.arange(len(valid_labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, baseline_values, width, label='Baseline', alpha=0.8, color='steelblue')
    bars2 = ax.bar(x + width/2, tuned_values, width, label='Tuned', alpha=0.8, color='darkorange')
    
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Baseline vs Tuned Model Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(valid_labels)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.0])
    
    # Add value labels on bars
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
    
    autolabel(bars1)
    autolabel(bars2)
    
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
    
    print(f"Original features: {original_features}")
    print(f"Feature selection method: {args.feature_selection}")
    
    if args.feature_selection == 'lasso':
        print(f"\nLasso (L1) Feature Selection")
        
        start_time = time.time()
        lasso_indices, lasso_features, lasso_selector, n_lasso = select_features_lasso(
            X, y, C=args.lasso_C, random_state=args.random_seed
        )
        lasso_time = time.time() - start_time
        
        X = X.iloc[:, lasso_indices]
        selected_feature_names = lasso_features
        
        print(f"  Selection time: {lasso_time:.2f} seconds")
        print(f"\n✓ Using Lasso-selected features for training")
        print(f"Final feature set: {X.shape[1]} features")
        print(f"Speed improvement: ~{original_features / X.shape[1]:.1f}x faster!\n")
    else:  # none
        print(f"\n✓ No feature selection - using all {original_features} features")
        selected_feature_names = X.columns.tolist()
        print(f"  Note: This will be slower but allows comparison with/without feature selection\n")
    
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
    y_train_proba = rf.predict_proba(X_train)
    train_metrics = evaluate_classifier(y_train, y_train_pred, y_train_proba, target_names)
    
    # Evaluate on test set
    print("\n" + "=" * 70)
    print("TEST SET EVALUATION")
    print("=" * 70)
    y_test_pred = rf.predict(X_test)
    y_test_proba = rf.predict_proba(X_test)
    test_metrics = evaluate_classifier(y_test, y_test_pred, y_test_proba, target_names)
    
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
    
    # Hyperparameter Tuning (if enabled)
    tuned_train_metrics = None
    tuned_test_metrics = None
    best_rf = None
    
    if args.tune_hyperparameters:
        # Store baseline metrics
        baseline_train_metrics = train_metrics.copy()
        baseline_test_metrics = test_metrics.copy()
        
        # Perform hyperparameter tuning
        best_rf, search_results = tune_hyperparameters(
            X_train, y_train,
            n_iter=args.tuning_iterations,
            cv_folds=args.cv_folds,
            random_state=args.random_seed
        )
        
        # Evaluate tuned model
        print("\n" + "=" * 70)
        print("TUNED MODEL EVALUATION")
        print("=" * 70)
        
        # Training set
        print("\nTuned Model - Training Set:")
        y_train_pred_tuned = best_rf.predict(X_train)
        y_train_proba_tuned = best_rf.predict_proba(X_train)
        tuned_train_metrics = evaluate_classifier(y_train, y_train_pred_tuned, y_train_proba_tuned, target_names)
        
        # Test set
        print("\nTuned Model - Test Set:")
        y_test_pred_tuned = best_rf.predict(X_test)
        y_test_proba_tuned = best_rf.predict_proba(X_test)
        tuned_test_metrics = evaluate_classifier(y_test, y_test_pred_tuned, y_test_proba_tuned, target_names)
        
        # Compare models
        compare_models(baseline_test_metrics, tuned_test_metrics, dataset_name='Test')
        
        # Discussion of tuning impact
        print("\n" + "=" * 70)
        print("IMPACT OF HYPERPARAMETER TUNING")
        print("=" * 70)
        
        # Calculate improvements
        acc_improvement = tuned_test_metrics['accuracy'] - baseline_test_metrics['accuracy']
        f1_improvement = tuned_test_metrics['f1_score'] - baseline_test_metrics['f1_score']
        auc_improvement = (tuned_test_metrics.get('auc_roc', 0) - baseline_test_metrics.get('auc_roc', 0)) if baseline_test_metrics.get('auc_roc') else 0
        
        print("\nDiscussion:")
        if acc_improvement > 0.001:  # More than 0.1% improvement
            print(f"  ✓ Accuracy improved by {acc_improvement:.4f} ({acc_improvement*100:.2f}%)")
            print(f"    This means {int(acc_improvement * len(y_test))} more correct predictions on the test set")
        
        if f1_improvement > 0.001:
            print(f"  ✓ F1-score improved by {f1_improvement:.4f} ({f1_improvement*100:.2f}%)")
            print(f"    Better balance between precision and recall")
        
        if auc_improvement > 0.001:
            print(f"  ✓ AUC-ROC improved by {auc_improvement:.4f}")
            print(f"    Better discrimination between classes")
        
        best_params = search_results.best_params_
        print(f"\nKey findings from tuning:")
        
        if best_params.get('max_depth') is not None:
            print(f"  • Limited tree depth to {best_params['max_depth']} reduces overfitting")
        else:
            print(f"  • Unlimited tree depth works best for this dataset")
        
        print(f"  • Optimal number of trees: {best_params.get('n_estimators', 'N/A')}")
        print(f"  • Best split criterion: {best_params.get('criterion', 'N/A')}")
        print(f"  • Optimal features per split: {best_params.get('max_features', 'N/A')}")
        print(f"  • min_samples_split={best_params.get('min_samples_split', 2)} controls node splitting")
        
        print(f"\nGeneralization check:")
        train_test_gap_baseline = baseline_train_metrics['accuracy'] - baseline_test_metrics['accuracy']
        train_test_gap_tuned = tuned_train_metrics['accuracy'] - tuned_test_metrics['accuracy']
        
        print(f"  Baseline train-test gap: {train_test_gap_baseline:.4f}")
        print(f"  Tuned train-test gap:    {train_test_gap_tuned:.4f}")
        
        if train_test_gap_tuned < train_test_gap_baseline:
            print(f"  ✓ Reduced overfitting! Better generalization to unseen data")
        else:
            print(f"  ○ Train-test gap similar or larger")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if args.tune_hyperparameters and tuned_test_metrics:
        print("\n=== BASELINE MODEL ===")
    
    print(f"\nTraining Accuracy:    {train_metrics['accuracy']:.4f}")
    print(f"Testing Accuracy:     {test_metrics['accuracy']:.4f}")
    print(f"CV Mean Accuracy:     {mean_cv_score:.4f} (+/- {std_cv_score:.4f})")
    print(f"\nTesting Metrics:")
    print(f"  F1 Score:           {test_metrics['f1_score']:.4f}")
    print(f"  Precision:          {test_metrics['precision']:.4f}")
    print(f"  Recall:             {test_metrics['recall']:.4f}")
    if 'auc_roc' in test_metrics and test_metrics['auc_roc'] is not None:
        print(f"  AUC-ROC:            {test_metrics['auc_roc']:.4f}")
    
    if args.tune_hyperparameters and tuned_test_metrics:
        print("\n=== TUNED MODEL ===")
        print(f"\nTraining Accuracy:    {tuned_train_metrics['accuracy']:.4f}")
        print(f"Testing Accuracy:     {tuned_test_metrics['accuracy']:.4f}")
        print(f"\nTesting Metrics:")
        print(f"  F1 Score:           {tuned_test_metrics['f1_score']:.4f}")
        print(f"  Precision:          {tuned_test_metrics['precision']:.4f}")
        print(f"  Recall:             {tuned_test_metrics['recall']:.4f}")
        if 'auc_roc' in tuned_test_metrics and tuned_test_metrics['auc_roc'] is not None:
            print(f"  AUC-ROC:            {tuned_test_metrics['auc_roc']:.4f}")
    
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
    
    # ROC curve for test set
    print("\nGenerating ROC curve for test set...")
    plot_roc_curve(
        y_test, y_test_proba,
        title=f'ROC Curve - Test Set (AUC: {test_metrics.get("auc_roc", 0):.4f})',
        save_path='roc_curve_test.png',
        target_names=target_names
    )
    print("  Saved ROC curve to roc_curve_test.png")
    
    # ROC curve for training set
    print("\nGenerating ROC curve for training set...")
    plot_roc_curve(
        y_train, y_train_proba,
        title=f'ROC Curve - Training Set (AUC: {train_metrics.get("auc_roc", 0):.4f})',
        save_path='roc_curve_train.png',
        target_names=target_names
    )
    print("  Saved ROC curve to roc_curve_train.png")
    
    # If tuning was performed, generate comparison visualizations
    if args.tune_hyperparameters and tuned_test_metrics:
        print("\nGenerating tuned model visualizations...")
        
        # Confusion matrix for tuned model
        plot_confusion_matrix(
            y_test, y_test_pred_tuned,
            title=f'Confusion Matrix - Tuned Model (Accuracy: {tuned_test_metrics["accuracy"]:.4f})',
            save_path='confusion_matrix_tuned.png',
            target_names=target_names
        )
        print("  Saved tuned model confusion matrix to confusion_matrix_tuned.png")
        
        # ROC curve for tuned model
        plot_roc_curve(
            y_test, y_test_proba_tuned,
            title=f'ROC Curve - Tuned Model (AUC: {tuned_test_metrics.get("auc_roc", 0):.4f})',
            save_path='roc_curve_tuned.png',
            target_names=target_names
        )
        print("  Saved tuned model ROC curve to roc_curve_tuned.png")
        
        # Comparison plot
        print("\nGenerating baseline vs tuned comparison plot...")
        plot_tuning_comparison(baseline_test_metrics, tuned_test_metrics, save_path='model_comparison.png')
        print("  Saved model comparison plot to model_comparison.png")
    
    print("\n" + "=" * 70)
    print("CLASSIFICATION ANALYSIS COMPLETE!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - confusion_matrix_test.png")
    print("  - confusion_matrix_train.png")
    print("  - roc_curve_test.png")
    print("  - roc_curve_train.png")
    print("  - cv_scores.png")
    
    if args.tune_hyperparameters and tuned_test_metrics:
        print("\n  Hyperparameter Tuning Results:")
        print("  - confusion_matrix_tuned.png")
        print("  - roc_curve_tuned.png")
        print("  - model_comparison.png")
    
    print()


if __name__ == '__main__':
    main()

