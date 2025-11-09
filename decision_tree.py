from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import pandas as pd
from node import Node

class DecisionTree(object):
    """
    Optimized CART-style decision tree with efficient numeric and categorical splitting.
    Supports both Gini and Entropy criteria with fast threshold search.
    """
    
    def __init__(self, criterion: Optional[str] = 'gini',
                 max_depth: Optional[int] = None,
                 min_samples_split: Optional[int] = None):
        """
        :param criterion: 'gini' for Gini impurity or 'entropy' for information gain
        :param max_depth: Maximum depth of the tree
        :param min_samples_split: Minimum samples required to split a node
        """
        if criterion not in ('gini', 'entropy'):
            raise ValueError("criterion must be 'gini' or 'entropy'")
        
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = max(2, min_samples_split) if min_samples_split else 2
        self.tree = None
        self._feature_types: Dict[str, str] = {}
        self._categorical_values: Dict[str, List] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> float:
        """
        Build decision tree and return training accuracy.
        """
        # Reset indices for consistency
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        
        # Infer feature types and cache categorical values once
        self._feature_types.clear()
        self._categorical_values.clear()
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                self._feature_types[col] = 'num'
            else:
                self._feature_types[col] = 'cat'
                # Cache unique categorical values for efficiency
                self._categorical_values[col] = X[col].dropna().unique().tolist()
        
        # Build tree
        self.tree = self._build_tree(X, y, depth=0)
        return self.evaluate(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict classes for all rows in X.
        """
        predictions = []
        for i in range(len(X)):
            row = X.iloc[i]
            node = self.tree
            while not node.is_leaf:
                child = node.get_child_node(row[node.name])
                if child is None:
                    break
                node = child
            predictions.append(node.node_class)
        return np.array(predictions)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> float:
        """
        Calculate accuracy of predictions.
        """
        preds = self.predict(X)
        return float(np.mean(preds == y.to_numpy()))

    def gini(self, X: pd.DataFrame, y: pd.Series, feature: str) -> float:
        """
        Calculate Gini impurity for a feature.
        """
        if self._feature_types[feature] == 'num':
            result = self._best_numeric_split(X, feature, y)
        else:
            result = self._best_categorical_split(X, feature, y)
        return result[0] if result else float('inf')

    def entropy(self, X: pd.DataFrame, y: pd.Series, feature: str) -> float:
        """
        Calculate entropy for a feature.
        """
        if self._feature_types[feature] == 'num':
            result = self._best_numeric_split(X, feature, y)
        else:
            result = self._best_categorical_split(X, feature, y)
        return result[0] if result else float('inf')

    # Core tree building 
    def _build_tree(self, X: pd.DataFrame, y: pd.Series, depth: int) -> Node:
        """Build tree recursively with optimized stopping conditions."""
        node_class = self._majority_class(y)
        node = Node(node_size=len(y), node_class=node_class, depth=depth, 
                   single_class=self._is_pure(y))
        
        # Stopping conditions
        if (self._is_pure(y) or 
            (self.max_depth is not None and depth >= self.max_depth) or
            len(y) < self.min_samples_split or
            X.empty):
            return node
        
        # Find best split
        best_split = self._find_best_split(X, y)
        if best_split is None:
            return node
        
        feat, is_numeric, threshold, partitions = best_split
        
        # Create children
        children = {}
        if is_numeric:
            for key, (Xi, yi) in [('l', partitions['l']), ('ge', partitions['ge'])]:
                children[key] = self._build_tree(Xi, yi, depth + 1)
            node.name = feat
            node.is_numerical = True
            node.threshold = threshold
            node.set_children(children)
        else:
            for val, (Xi, yi) in partitions.items():
                children[val] = self._build_tree(Xi, yi, depth + 1)
            node.name = feat
            node.is_numerical = False
            node.set_children(children)
        
        return node

    def _find_best_split(self, X: pd.DataFrame, y: pd.Series) -> Optional[Tuple]:
        """Find the best split across all features."""
        # Early stopping: if already pure, no need to search for splits
        if self._is_pure(y):
            return None
            
        base_impurity = self._impurity(y)
        best_gain = -np.inf
        best_split = None
        
        for feat in X.columns:
            if self._feature_types[feat] == 'num':
                result = self._best_numeric_split(X, feat, y)
            else:
                result = self._best_categorical_split(X, feat, y)
            
            if result is None:
                continue
                
            imp_after, threshold, partitions = result
            gain = base_impurity - imp_after
            
            if gain > best_gain and len(partitions) >= 2:
                best_gain = gain
                is_numeric = self._feature_types[feat] == 'num'
                best_split = (feat, is_numeric, threshold, partitions)
        
        return best_split

    # Impurity calculations 
    def _impurity(self, y: pd.Series) -> float:
        """Calculate impurity based on criterion."""
        return self._gini_impurity(y) if self.criterion == 'gini' else self._entropy_impurity(y)

    def _gini_impurity(self, y: pd.Series) -> float:
        """Fast Gini impurity calculation."""
        if len(y) == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        p = counts / counts.sum()
        return float(1.0 - np.sum(p * p))

    def _entropy_impurity(self, y: pd.Series) -> float:
        """Fast entropy calculation."""
        if len(y) == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        p = counts / counts.sum()
        eps = 1e-12  # Avoid log(0)
        return float(-np.sum(p * np.log2(p + eps)))

    # Split optimization 
    def _best_numeric_split(self, X: pd.DataFrame, feat: str, y: pd.Series) -> Optional[Tuple]:
        """Optimized numeric split with prefix sums for fast impurity calculation."""
        x = X[feat]
        # Remove missing values for threshold search
        valid_mask = ~x.isna()
        if not valid_mask.any():
            return None
            
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]
        
        # Sort by feature values
        sorted_indices = x_valid.argsort(kind='mergesort')
        x_sorted = x_valid.iloc[sorted_indices].values
        y_sorted = y_valid.iloc[sorted_indices].values
        
        # Get unique classes and create mapping
        classes = np.unique(y_sorted)
        class_to_idx = {c: i for i, c in enumerate(classes)}
        n_classes = len(classes)
        n_samples = len(x_sorted)
        
        # Build prefix sum for fast impurity calculation
        prefix_counts = np.zeros((n_samples, n_classes), dtype=np.int64)
        for i, label in enumerate(y_sorted):
            if i > 0:
                prefix_counts[i] = prefix_counts[i-1]
            prefix_counts[i, class_to_idx[label]] += 1
        
        best_impurity = np.inf
        best_threshold = None
        
        # Check thresholds where class changes
        for i in range(1, n_samples):
            if x_sorted[i] == x_sorted[i-1] or y_sorted[i] == y_sorted[i-1]:
                continue
                
            left_counts = prefix_counts[i-1]
            right_counts = prefix_counts[-1] - left_counts
            n_left = left_counts.sum()
            n_right = right_counts.sum()
            
            if n_left == 0 or n_right == 0:
                continue
            
            # Calculate weighted impurity
            if self.criterion == 'gini':
                p_left = left_counts / n_left
                p_right = right_counts / n_right
                imp_left = 1.0 - np.sum(p_left * p_left)
                imp_right = 1.0 - np.sum(p_right * p_right)
            else:
                eps = 1e-12
                p_left = left_counts / n_left
                p_right = right_counts / n_right
                imp_left = -np.sum(p_left * np.log2(p_left + eps))
                imp_right = -np.sum(p_right * np.log2(p_right + eps))
            
            weighted_impurity = (n_left * imp_left + n_right * imp_right) / (n_left + n_right)
            
            if weighted_impurity < best_impurity:
                best_impurity = weighted_impurity
                best_threshold = (x_sorted[i-1] + x_sorted[i]) / 2.0
        
        if best_threshold is None:
            return None
        
        # Create partitions
        left_mask = x < best_threshold
        right_mask = x >= best_threshold
        partitions = {
            'l': (X[left_mask], y[left_mask]),
            'ge': (X[right_mask], y[right_mask])
        }
        
        return best_impurity, best_threshold, partitions

    def _best_categorical_split(self, X: pd.DataFrame, feat: str, y: pd.Series) -> Optional[Tuple]:
        """Optimized categorical split with vectorized operations."""
        x = X[feat]
        # Handle missing values as separate category
        x_filled = x.astype('object').fillna('__MISSING__')
        
        # Group by category values
        groups = x_filled.groupby(x_filled)
        if len(groups) <= 1:
            return None
        
        total_n = len(y)
        classes = np.unique(y)
        class_to_idx = {c: i for i, c in enumerate(classes)}
        
        weighted_impurity = 0.0
        partitions = {}
        
        for val, group_indices in groups.groups.items():
            # Use boolean indexing instead of iloc to avoid index issues
            mask = (x_filled == val)
            yi = y[mask]
            Xi = X[mask]
            n_group = len(yi)
            
            if n_group == 0:
                continue
            
            # Count classes in this group
            counts = np.zeros(len(classes), dtype=np.int64)
            for label, cnt in yi.value_counts().items():
                counts[class_to_idx[label]] = cnt
            
            # Calculate impurity for this group
            if self.criterion == 'gini':
                p = counts / counts.sum()
                group_impurity = 1.0 - np.sum(p * p)
            else:
                eps = 1e-12
                p = counts / counts.sum()
                group_impurity = -np.sum(p * np.log2(p + eps))
            
            weighted_impurity += (n_group / total_n) * group_impurity
            partitions[val] = (Xi, yi)
        
        return weighted_impurity, None, partitions

    #  Utility methods
    def _majority_class(self, y: pd.Series) -> Any:
        """Get majority class with deterministic tie-breaking."""
        counts = y.value_counts()
        max_count = counts.max()
        winners = sorted(counts[counts == max_count].index.tolist(), key=str)
        return winners[0]

    def _is_pure(self, y: pd.Series) -> bool:
        """Check if all labels are the same."""
        return y.nunique() <= 1

