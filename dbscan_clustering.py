import numpy as np

class DBScan:
    """
    Density-based clustering with noise detection.
    """

    def __init__(self, eps: float = 1.0, minPts: int = 5, heuristic: str = "euclidean"):
        """
        :param eps: radius of each neighbourhood
        :param minPts: number of points in a neighbourhood
        :param heuristic: measures euclidean distance between points 
        """ 
        self.eps = eps 
        self.minPts = minPts
        self.heuristic = heuristic
        self.labels = None # Number of points that have been labeled (core, border, noise)
        self.n_clusters = None # Number of clusters formed from core points 
    
    def euclidean_distance(self, P1: np.ndarray, P2: np.ndarray):
        """
        Compute euclidean distance between 2 points P1 and P2 in neighbourhood
        """
        P1_sq = np.sum(P1 ** 2, axis = 1).reshape(-1, 1)
        P2_sq = np.sum(P2 ** 2, axis = 0)
        dist_sq = P1_sq + P2_sq - 2 * P1 @ P1.T 
        dist_sq = np.maximum(dist_sq, 0.0)
        return np.sqrt(dist_sq)
    
    def fit(self, X: np.ndarray):
        """
        Performs density-based clustering on X
        :param X: array of shape (n_samples, n_features)

        labels = -1 indicates that it has not been labelled
                  0 indicates that it is a noise point 
        """
        n_samples = X.shape[0]
        cid = 0 # current cluster/core pt 
        clusters = []
        labels = np.zeros(n_samples) - 1 
        

        # Iterate through every point and assign as core pt, border pt or noise pt
        for i in range(0, n_samples):
            # Start with no points scanned
            if labels[i] == -1:
                # Find neighbours of current point within epsilon radius
                neighbours = self.getNeighbours(X, i)
                if len(neighbours) < self.minPts: # Neighbouring pts are scattered 
                    labels[i] = 0 # set as noise pt
                else: # if current point is core pt 
                    cid = cid + 1 
                    labels[i] = cid
                    clusters.append(i)
                    for neighbor in neighbours:
                        if labels[neighbor] == -1: # Neighbouring point has not been labelled
                            labels[neighbor] = cid
                            prev_cid = cid - 1 
                            clusters[prev_cid] = neighbor
                            add_neighbours = self.getNeighbours(X, neighbor)
                            if len(add_neighbours) >= self.minPts: 
                                neighbours = neighbours + add_neighbours # add new neighbours to cluster
                        elif labels[neighbor] == 0: # Neighbouring points is noise 
                            labels[neighbor] = cid
                            prev_cid = cid - 1 
                            clusters[prev_cid] = neighbor

        self.labels = labels
        self.n_clusters = clusters 
        return clusters 
    
    def getNeighbours(self, X: np.ndarray, pt: np.ndarray):
        # Add: check for uniqueness of each neighbouring point
        neighbours = []
        n_samples = X.shape[0]
        for i in range(0, n_samples):
            dist = self.euclidean_distance(X, pt)
        if dist.all() <= self.eps: # if other pt is in radius
            neighbours.append(i) # add to potential cluster

        return neighbours 










                










