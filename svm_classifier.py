"""
Support Vector Machine Classifier for Diabetic Readmission
CMPT 459 Course Project

The following script trains a SVM classifier, evaluates on a train/test split,
and prints evaluation metrics. 

Adjusted to match structure of knn_classifier.py and classification_analysis.py
"""
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)

class SVMClassifier:
    """ Trains a soft SVM classifer for non-linearly separable datasets without kernel."""

    def __init__(self, alpha: float = 0.001, lmda: float = 0.01, num_iterations: int = 100):
        """
        :param alpha: learning rate of SVM
        :param lmda: tradeoff between margin size and x_i inside margin
        :param num_iterations: number of iterations 
        """
        self.alpha = alpha
        self.lmda = lmda
        self.num_iterations = num_iterations
        self.b = None # Intercept of hyperplane equation
        self.weights = None # Weights of training object
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """:param X: array of shape (n_samples, n_features)"""
        n_samples, n_features = X.shape

        # Initialize both as 0 
        weights = np.zeros(n_features)
        b = 0 
        iterations = self.num_iterations 

        for iter in range(iterations):
            # Update weights and intercept based on margin 
            for i, X_i in enumerate(X):
                # Points lie within slack margin
                if y[i] * (np.dot(weights, X_i) - b) >= 1: # Case: y_i * w_i * x_i - b >= 1
                    # Update weight by the following: 
                    # new weight = weight + alpha * (2lambda * weight - y_i * x_i)
                    weights = weights - (self.alpha * (2 * self.lmda * weights))

                else: # Case: y_i * w_i * x_i - b < 1
                    # Update weight as: weight + alpha * (2lambda * weight - y_i * x_i)
                    weights = weights - (self.alpha * (2 * self.lmda * weights - np.dot(y[i], X_i)))
                    # Update intercept b as: new intercept = b - alpha * (y_i)
                    b = b - (self.alpha * y[i])
            
            self.weights = weights
            self.b = b 
             
            return weights, b
    
    def predict(self, X: np.ndarray):
        prediction = np.dot(X, self.weights) - self.b
        # Returns an array of binary results (-1,1)
        pred_result = [1 if pred > 0 else -1 for pred in prediction]
        return pred_result 
    
def getHyperplane(X: np.ndarray, weights, b, offset):
    """ Helper function for visualization of hyperplanes."""
    # Hyperplane equation: X_i * W + b = 0
    # Draws a plane with soft margins  
    hyperplane = (-weights[0] * X + b + offset) / weights[1]
    return hyperplane

# Preprocess data (same as knn_classifier.py)
def load_and_preprocess_data(path: str):

    print("Loading data...")
    df = pd.read_csv(path)
    print(f"Original shape: {df.shape}")

    df = df.replace('?', np.nan)

    # Drop columns with >40% missing
    threshold = 0.4 * len(df)
    df = df.dropna(thresh=threshold, axis=1)

    # Fill categorical NAs
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna("Unknown")

    # Encode target
    df["readmitted"] = df["readmitted"].map({'NO': 0, '>30': 1, '<30': 2})

    # Encode categorical
    cat_cols = df.select_dtypes(include='object').columns
    le = LabelEncoder()
    for col in cat_cols:
        if df[col].nunique() < 10:
            df[col] = le.fit_transform(df[col].astype(str))
        else:
            df = pd.get_dummies(df, columns=[col], drop_first=True)

    # Drop IDs
    for col in ["encounter_id", "patient_nbr"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Scale numeric
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    X = df.drop(columns=["readmitted"]).values
    y = df["readmitted"].values
    print("Preprocessing complete! Final shape:", X.shape)
    return X, y


def plot_svm(X, label_true, label_pred, save_path):
    """ Visualization of the hyperplane that separates 2 classes.
        To do: Superimpose hyperplane and predictions onto test sample plot
    """    

    print("Reducing data via PCA for 2D Visualization")
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(10, 8))
    plt.title("Plot for linear SVM Classification", fontsize = 14, fontweight = "bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi = 300, bbox_inches = "tight")
    plt.close()

# Parse arguments to run SVM Classifier
def parse_args():
    parser = argparse.ArgumentParser(description = 'SVM Classifier for Diabetic Readmission')
    parser.add_argument('--data', type = str, default = 'data/diabetic_data.csv',
                        help = 'Path to diabetic dataset CSV')
    parser.add_argument('--test-size', type = float, default = 0.2,
                        help='Test set to training set proportion')
    parser.add_argument('--alpha', type = float, default = 0.001,
                        help = 'Set learning rate')
    parser.add_argument('--lmda', type = float, default = 0.01,
                        help='Margin size tradeoff')
    parser.add_argument('--iterations', type = int, default = 100,
                        help='Number of iterations')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()

def main():
    """ Main script to run SVM classifier on the diabetic dataset. """
    args = parse_args()
    np.random.seed(args.random_state)

    print("=" * 70)
    print("Soft SVM Classifier")
    print("=" * 70)
    print(f"Dataset:      {args.data}")
    print(f"Learning rate (alpha): {args.alpha}")
    print(f"Margin tradeoff (lambda): {args.lmda}")
    print(f"Number of iterations: {args.iterations}")
    print("=" * 70)

    # Preprocess data for classification
    X, y = load_and_preprocess_data(args.data)
    # Create train/test split of data 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = args.test_size, random_state = args.random_state, stratify = y
    )

    # Train Soft SVM
    svm = SVMClassifier(
        alpha = args.alpha,
        lmda = args.lmda,
        num_iterations = args.iterations
    )
    svm.fit(X_train, y_train)
    
    # Get classification predictions 
    label_pred = svm.predict(X_test)
 
    # Evaluation Metrics
    accuracy = accuracy_score(y_test, label_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, label_pred, average = 'weighted', zero_division = 0
    )

    print("\nClassification Results:")
    print("=" * 70)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-score: {f1:.4f}")

    # PCA visualization
    save_file = f"svm_pca.png"
    plot_svm(X_test, y_test, label_pred, save_file)
    print("Results saved as ", save_file)
    print("=" * 70)

if __name__ == '__main__':
    main()






