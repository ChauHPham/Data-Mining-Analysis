import numpy as np


class KMeans():

    def __init__(self, n_clusters: int, init: str='kmeans++', max_iter = 300):
        """

        :param n_clusters: number of clusters
        :param init: centroid initialization method. Should be 'kmeans++'
        :param max_iter: maximum number of iterations
        """
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.centroids = None # Initialized in initialize_centroids()

    def fit(self, X: np.ndarray):
        self.initialize_centroids(X)
        clustering = np.zeros(X.shape[0])
        
        for iteration in range(self.max_iter):
            # Assign each point to the nearest centroid
            # Compute distances from all points to all centroids
            distances = self.euclidean_distance(X, self.centroids)  # (n_samples, n_clusters)
            
            # Assign each point to the cluster with the nearest centroid
            new_clustering = np.argmin(distances, axis=1)
            
            # Check for convergence (if clustering doesn't change, stop)
            if np.array_equal(clustering, new_clustering):
                break
            
            clustering = new_clustering
            
            # Update centroids based on the new clustering
            self.update_centroids(clustering, X)
        
        return clustering

    def update_centroids(self, clustering: np.ndarray, X: np.ndarray):
        """
        Updates centroids based on the current clustering.
        Each centroid is set to the mean of all points assigned to that cluster.
        Optimized version using vectorized operations.
        """
        n_features = X.shape[1]
        new_centroids = np.zeros((self.n_clusters, n_features))
        
        for k in range(self.n_clusters):
            # Find all points assigned to cluster k
            mask = (clustering == k)
            if np.sum(mask) > 0:
                # Update centroid to mean of points in cluster k
                new_centroids[k] = X[mask].mean(axis=0)
            else:
                # If no points assigned to cluster k, keep the previous centroid
                new_centroids[k] = self.centroids[k]
        
        self.centroids = new_centroids

    def initialize_centroids(self, X: np.ndarray):
        """
        Initialize centroids using kmeans++ method of initialization.
        :param X:
        :return:
        """
        n_samples, n_features = X.shape
        
        # KMeans++ initialization
        centroids = np.zeros((self.n_clusters, n_features))
        
        # Step 1: Randomly select the first centroid
        first_idx = np.random.randint(n_samples)
        centroids[0] = X[first_idx].copy()
        
        # Steps 2-3: Select remaining centroids
        for k in range(1, self.n_clusters):
            # Compute distance from each point to the nearest chosen centroid
            # centroids[:k] contains all previously chosen centroids
            distances = self.euclidean_distance(X, centroids[:k])  # (n_samples, k)
            min_distances = np.min(distances, axis=1)  # (n_samples,)
            
            # Compute probabilities (proportional to squared distance)
            probabilities = min_distances ** 2
            probabilities = probabilities / probabilities.sum()
            
            # Select next centroid based on probabilities
            next_idx = np.random.choice(n_samples, p=probabilities)
            centroids[k] = X[next_idx].copy()
        
        self.centroids = centroids

    def euclidean_distance(self, X1:np.ndarray, X2:np.ndarray):
        """
        Computes the euclidean distance between all pairs (x,y) where x is a row in X1 and y is a row in X2.
        Optimized version using matrix multiplication: ||x-y||^2 = ||x||^2 + ||y||^2 - 2xy^T
        """
        # Optimized using: ||x-y||^2 = ||x||^2 + ||y||^2 - 2xy^T
        X1_sq = np.sum(X1**2, axis=1).reshape(-1, 1)  # (n1, 1)
        X2_sq = np.sum(X2**2, axis=1)  # (n2,)
        dist_sq = X1_sq + X2_sq - 2 * X1 @ X2.T  # (n1, n2)
        # Handle numerical precision issues
        dist_sq = np.maximum(dist_sq, 0)
        dist = np.sqrt(dist_sq)
        return dist

    def silhouette(self, clustering: np.ndarray, X: np.ndarray, sample_size: int = None):
        """
        Computes the silhouette coefficient for the clustering.
        Optimized version using vectorization and sampling for very large datasets.
        For each point i:
        - a(i): average distance from point i to all other points in the same cluster
        - b(i): average distance from point i to all points in the nearest neighboring cluster
        - s(i) = (b(i) - a(i)) / max(a(i), b(i))
        Returns the mean silhouette coefficient across all points.
        """
        n_samples = X.shape[0]
        
        # Use sampling if dataset is very large
        if sample_size is None:
            # Sample up to 5000 points for silhouette computation if dataset is large
            sample_size = min(5000, n_samples) if n_samples > 5000 else n_samples
        
        if sample_size < n_samples:
            # Sample random points for faster computation
            sample_indices = np.random.choice(n_samples, size=sample_size, replace=False)
            X_sample = X[sample_indices]
            clustering_sample = clustering[sample_indices]
        else:
            X_sample = X
            clustering_sample = clustering
            sample_indices = np.arange(n_samples)
        
        n_sample = X_sample.shape[0]
        unique_clusters = np.unique(clustering_sample)
        
        # Pre-compute distances from sample points to all points in X
        # This is fully vectorized - done in one operation
        distances_to_all = self.euclidean_distance(X_sample, X)  # (n_sample, n_samples)
        
        # Vectorized computation of a(i) and b(i) for all sample points
        a_values = np.zeros(n_sample)
        b_values = np.zeros(n_sample)
        
        for i in range(n_sample):
            cluster_i = int(clustering_sample[i])
            
            # a(i): average distance from point i to all other points in the same cluster
            # Vectorized: use boolean masking
            same_cluster_mask = (clustering == cluster_i)
            same_cluster_mask[sample_indices[i]] = False  # Exclude point i itself
            same_cluster_distances = distances_to_all[i, same_cluster_mask]
            
            if len(same_cluster_distances) == 0:
                a_values[i] = 0
                b_values[i] = 0
                continue
            
            a_i = np.mean(same_cluster_distances)
            a_values[i] = a_i
            
            # b(i): average distance from point i to all points in the nearest neighboring cluster
            # Vectorized: compute mean distance to each cluster, then take min
            cluster_means = []
            for k in unique_clusters:
                if k != cluster_i:
                    cluster_k_mask = (clustering == k)
                    if np.any(cluster_k_mask):
                        # Vectorized mean computation
                        cluster_mean = np.mean(distances_to_all[i, cluster_k_mask])
                        cluster_means.append(cluster_mean)
            
            if len(cluster_means) > 0:
                b_i = np.min(cluster_means)  # Vectorized min
            else:
                b_i = a_i
            b_values[i] = b_i
        
        # Vectorized computation of silhouette scores for all points
        # Handle edge cases vectorized
        max_ab = np.maximum(a_values, b_values)
        non_zero_mask = max_ab > 0
        silhouette_scores = np.zeros(n_sample)
        silhouette_scores[non_zero_mask] = (b_values[non_zero_mask] - a_values[non_zero_mask]) / max_ab[non_zero_mask]
        
        # Return the mean silhouette coefficient
        return np.mean(silhouette_scores)

