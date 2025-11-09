"""
Random Forest Classifier
CMPT 459 Course Project

A Random Forest implementation using bootstrap aggregation (bagging) of Decision Trees.
Each tree is trained on a bootstrap sample with random feature subset selection.
"""

import numpy as np
import pandas as pd
from typing import Optional
from decision_tree import DecisionTree


class RandomForest:
    """
    Random Forest classifier using Decision Trees as base learners.
    
    Implements bootstrap aggregation (bagging) where:
    - Each tree is trained on a bootstrap sample (sampling with replacement)
    - At each split, only a random subset of features is considered
    - Final prediction is majority vote across all trees
    """
    
    def __init__(self,
                 n_estimators: int = 100,
                 criterion: str = 'gini',
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 2,
                 max_features: Optional[str] = 'sqrt',
                 random_state: Optional[int] = None):
        """
        Initialize Random Forest.
        
        :param n_estimators: Number of trees in the forest
        :param criterion: Splitting criterion ('gini' or 'entropy')
        :param max_depth: Maximum depth of each tree
        :param min_samples_split: Minimum samples required to split a node
        :param max_features: Number of features to consider at each split:
            - 'sqrt': sqrt(n_features)
            - 'log2': log2(n_features)
            - int: exact number
            - None: all features
        :param random_state: Random seed for reproducibility
        """
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []
        self.feature_indices = []  # Store which features each tree uses
        self.n_features_ = None
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'RandomForest':
        """
        Train the Random Forest.
        
        :param X: Training features (DataFrame)
        :param y: Training labels (Series)
        :return: self
        """
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        
        n_samples, n_features = X.shape
        self.n_features_ = n_features
        
        # Determine max_features to use at each split
        if self.max_features == 'sqrt':
            max_features = int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            max_features = int(np.log2(n_features))
        elif isinstance(self.max_features, int):
            max_features = min(self.max_features, n_features)
        elif self.max_features is None:
            max_features = n_features
        else:
            raise ValueError(f"Invalid max_features: {self.max_features}")
        
        # Build n_estimators trees
        self.trees = []
        self.feature_indices = []
        
        print(f"Training Random Forest with {self.n_estimators} trees...")
        for i in range(self.n_estimators):
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  Training tree {i + 1}/{self.n_estimators}...", end='\r', flush=True)
            
            # Bootstrap sample (sampling with replacement)
            bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_bootstrap = X.iloc[bootstrap_indices].reset_index(drop=True)
            y_bootstrap = y.iloc[bootstrap_indices].reset_index(drop=True)
            
            # Random feature subset
            feature_subset = np.random.choice(
                n_features, 
                size=max_features, 
                replace=False
            )
            feature_subset = sorted(feature_subset)  # Sort for consistency
            
            X_subset = X_bootstrap.iloc[:, feature_subset]
            
            # Train decision tree on bootstrap sample with feature subset
            tree = DecisionTree(
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )
            tree.fit(X_subset, y_bootstrap)
            
            self.trees.append(tree)
            self.feature_indices.append(feature_subset)
        
        print(f"\nRandom Forest training complete!")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict classes using majority voting across all trees.
        
        :param X: Test features (DataFrame)
        :return: Predicted classes
        """
        if len(self.trees) == 0:
            raise ValueError("Random Forest not trained. Call fit() first.")
        
        X = X.reset_index(drop=True)
        n_samples = len(X)
        
        # Collect predictions from all trees
        all_predictions = []
        for i, tree in enumerate(self.trees):
            feature_subset = self.feature_indices[i]
            X_subset = X.iloc[:, feature_subset]
            predictions = tree.predict(X_subset)
            all_predictions.append(predictions)
        
        # Stack predictions: (n_trees, n_samples)
        all_predictions = np.array(all_predictions)
        
        # Majority vote for each sample
        final_predictions = []
        for j in range(n_samples):
            votes = all_predictions[:, j]
            # Use mode (most frequent value)
            values, counts = np.unique(votes, return_counts=True)
            mode_index = np.argmax(counts)
            # In case of tie, choose first one (deterministic)
            final_predictions.append(values[mode_index])
        
        return np.array(final_predictions)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities (proportion of trees voting for each class).
        
        :param X: Test features (DataFrame)
        :return: Probability matrix of shape (n_samples, n_classes)
        """
        if len(self.trees) == 0:
            raise ValueError("Random Forest not trained. Call fit() first.")
        
        X = X.reset_index(drop=True)
        n_samples = len(X)
        
        # Collect predictions from all trees
        all_predictions = []
        for i, tree in enumerate(self.trees):
            feature_subset = self.feature_indices[i]
            X_subset = X.iloc[:, feature_subset]
            predictions = tree.predict(X_subset)
            all_predictions.append(predictions)
        
        # Stack predictions: (n_trees, n_samples)
        all_predictions = np.array(all_predictions)
        
        # Get unique classes
        unique_classes = np.unique(np.concatenate(all_predictions))
        unique_classes = np.sort(unique_classes)
        n_classes = len(unique_classes)
        
        # Calculate probabilities
        proba = np.zeros((n_samples, n_classes))
        for j in range(n_samples):
            votes = all_predictions[:, j]
            for k, cls in enumerate(unique_classes):
                proba[j, k] = np.sum(votes == cls) / self.n_estimators
        
        return proba
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> float:
        """
        Calculate accuracy on test set.
        
        :param X: Test features (DataFrame)
        :param y: Test labels (Series)
        :return: Accuracy score
        """
        predictions = self.predict(X)
        return float(np.mean(predictions == y.to_numpy()))

