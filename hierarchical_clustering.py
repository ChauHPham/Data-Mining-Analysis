import numpy as np


class HierarchicalClustering:
    """
    Simple agglomerative hierarchical clustering with average linkage.
    Designed for educational use; O(n^3) on full data, so use on a subset.
    """

    def __init__(self, n_clusters: int = 2, linkage: str = "average"):
        """
        :param n_clusters: desired number of clusters
        :param linkage: linkage type ('average' only for now)
        """
        assert linkage == "average", "Only 'average' linkage implemented."
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.labels_ = None  # cluster assignment for each point

    def _pairwise_distances(self, X: np.ndarray) -> np.ndarray:
        """
        Compute full pairwise Euclidean distance matrix.
        """
        X_sq = np.sum(X ** 2, axis=1).reshape(-1, 1)
        dist_sq = X_sq + X_sq.T - 2 * X @ X.T
        dist_sq = np.maximum(dist_sq, 0.0)
        return np.sqrt(dist_sq)

    def fit(self, X: np.ndarray):
        """
        Perform agglomerative clustering on X.

        :param X: array of shape (n_samples, n_features)
        :return: cluster labels as a 1D array of length n_samples
        """
        n_samples = X.shape[0]

        # Initial clusters: each point is its own cluster
        clusters = {i: [i] for i in range(n_samples)}

        # Precompute distance matrix between points
        point_dists = self._pairwise_distances(X)

        # Distance between clusters: we'll maintain a matrix where
        # dist_mat[i, j] is the average distance between all points in cluster i and j.
        # Start with identity of point distances.
        active_ids = list(clusters.keys())
        n_active = len(active_ids)
        dist_mat = point_dists.copy()

        # Use a large value on diagonal to avoid picking same cluster twice
        np.fill_diagonal(dist_mat, np.inf)

        while len(clusters) > self.n_clusters:
            # Find closest pair of clusters (i, j)
            # Only consider active cluster ids
            active_ids = list(clusters.keys())
            idx_map = {cid: idx for idx, cid in enumerate(active_ids)}
            submat = dist_mat[np.ix_(active_ids, active_ids)]

            # Ignore diagonal
            np.fill_diagonal(submat, np.inf)
            min_idx = np.unravel_index(np.argmin(submat), submat.shape)
            cid_i = active_ids[min_idx[0]]
            cid_j = active_ids[min_idx[1]]

            # Merge cluster j into i (arbitrary choice)
            new_members = clusters[cid_i] + clusters[cid_j]
            clusters[cid_i] = new_members
            del clusters[cid_j]

            # Update distances: average linkage
            for cid_k in clusters.keys():
                if cid_k == cid_i:
                    continue
                # average distance between all points in cluster i and k
                pts_i = clusters[cid_i]
                pts_k = clusters[cid_k]
                d_ik = point_dists[np.ix_(pts_i, pts_k)].mean()
                dist_mat[cid_i, cid_k] = d_ik
                dist_mat[cid_k, cid_i] = d_ik

            # Set deleted cluster row/col to inf so it's never chosen again
            dist_mat[cid_j, :] = np.inf
            dist_mat[:, cid_j] = np.inf

        # Assign labels
        labels = np.zeros(n_samples, dtype=int)
        for label, cid in enumerate(clusters.keys()):
            labels[clusters[cid]] = label

        self.labels_ = labels
        return labels
