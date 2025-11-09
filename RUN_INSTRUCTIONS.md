# How to Run the Code

## Prerequisites

Make sure you have the required Python packages installed:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## 1. Cluster Analysis

### Quick Test (Smaller k range for faster testing)
```bash
python cluster_analysis.py --min-k 2 --max-k 4 --pca-components 30
```

### Full Run (Default settings - Tests k=2 to k=6, 50 PCA components)
```bash
python cluster_analysis.py
```

### 3D Visualization (instead of 2D)
```bash
python cluster_analysis.py --vis-dims 3
```

### Custom Run
```bash
python cluster_analysis.py \
  --data data/diabetic_data.csv \
  --min-k 2 \
  --max-k 10 \
  --pca-components 100 \
  --vis-dims 2 \
  --random-seed 42
```

**Parameters:**
- `--data`: Path to CSV data file (default: data/diabetic_data.csv)
- `--min-k`: Minimum number of clusters to test (default: 2)
- `--max-k`: Maximum number of clusters to test (default: 6)
- `--pca-components`: Number of PCA components for clustering (default: 50)
- `--vis-dims`: Choose 2 or 3 for 2D/3D scatter plot visualization (default: 2)
- `--random-seed`: Random seed for reproducibility (default: 42)

**Optimizations Already in Place:**
- ✅ PCA reduces 2,389 features → 50 components
- ✅ Silhouette computed on 5,000 sampled points (not all 101,767!)
- ✅ KMeans++ initialization (fast & accurate)
- ✅ Vectorized distance computation

**Expected Output:**
- Console output showing silhouette scores for each k
- `silhouette_scores.png` - Line plot of silhouette scores vs k (KMeans++)
- `clustering_k*.png` - Best clustering visualization (KMeans++)
- `readmission_labels_pca.png` - Optional comparison with actual labels (2D only)

**Expected Runtime:** 
- Quick test (k=2-4, 30 PCA): ~20 seconds
- Full run (k=2-6, 50 PCA): ~30-60 seconds
- Extended run (k=2-10, 100 PCA): ~2-3 minutes

---

## 2. Classification Analysis

### Quick Test (Fewer trees and features for faster testing)
```bash
python classification_analysis.py --n-estimators 10 --n-features 50
```

### Full Run (Default settings - 20 trees, 100 features, depth 10)
```bash
python classification_analysis.py
```

### High Accuracy Run (More trees, features, and depth)
```bash
python classification_analysis.py --n-estimators 50 --n-features 200 --max-depth 15
```

### Custom Run
```bash
python classification_analysis.py \
  --data data/diabetic_data.csv \
  --test-size 0.2 \
  --n-estimators 50 \
  --max-depth 15 \
  --min-samples-split 2 \
  --criterion gini \
  --max-features sqrt \
  --cv-folds 5 \
  --random-seed 42
```

**Parameters:**
- `--n-features`: Number of best features to select (default: 100)
- `--n-estimators`: Number of trees (default: 20)
- `--max-depth`: Max tree depth (default: 10)
- `--cv-folds`: Cross-validation folds: 5 or 10 (default: 5)
- `--criterion`: Split criterion: gini or entropy (default: gini)
- `--max-features`: Features per split: sqrt, log2, or int (default: sqrt)
- `--test-size`: Test set proportion (default: 0.2)

**Feature Selection:**
- Automatically reduces 2,389 features to top N (default: 100)
- Uses ANOVA F-test to find most important features
- **~24x speed improvement!**

**Expected Output:**
- Console output showing training/test accuracy, cross-validation scores
- `confusion_matrix_test.png` - Confusion matrix for test set
- `confusion_matrix_train.png` - Confusion matrix for training set
- `cv_scores.png` - Cross-validation scores across folds
- `feature_importance.png` - Top 20 most important features

**Expected Runtime:** 
- Quick test (10 trees, 50 features): ~1-2 minutes
- Full run (20 trees, 100 features, depth 10): ~3-7 minutes
- High accuracy (50 trees, 200 features, depth 15): ~15-25 minutes

---

## Quick Verification

### Test if imports work:
```bash
python -c "from kmeans import KMeans; from random_forest import RandomForest; print('Imports OK')"
```

### Test cluster analysis (very quick, k=2 only):
```bash
python cluster_analysis.py --min-k 2 --max-k 2 --pca-components 10
```

### Test classification (very quick, 5 trees):
```bash
python classification_analysis.py --n-estimators 5 --cv-folds 3
```

---

## Troubleshooting

### If you get import errors:
```bash
# Make sure you're in the project directory
cd /Users/hoang/CMPT-459-Course-Project

# Check if files exist
ls -la *.py
```

### If you get file not found errors:
```bash
# Check if data file exists
ls -la data/diabetic_data.csv
```

### If you get memory errors:
- Reduce `--pca-components` (e.g., 50 instead of 100)
- Reduce `--n-estimators` (e.g., 50 instead of 100)
- Reduce `--max-k` range (e.g., 2-5 instead of 2-10)

