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
        return np.sqrt(np.sum((P1 - P2) ** 2))
    
    def fit(self, X: np.ndarray):
        """
        Performs density-based clustering on X
        :param X: array of shape (n_samples, n_features)

        labels = -1 indicates that it has not been labelled
                  0 indicates that it is a noise point 
        """
        n_samples = X.shape[0]
        cid = 0 # current cluster ID
        labels = np.zeros(n_samples) - 1  # -1 means unvisited
        

        # Iterate through every point and assign as core pt, border pt or noise pt
        for i in range(0, n_samples):
            # Skip if already processed
            if labels[i] != -1:
                continue
                
            # Find neighbours of current point within epsilon radius
            neighbours = self.getNeighbours(X, i)
            
            if len(neighbours) < self.minPts: # Neighbouring pts are scattered 
                labels[i] = 0 # set as noise pt
            else: # if current point is core pt 
                cid = cid + 1 
                labels[i] = cid
                
                # Expand cluster using a queue
                seed_set = list(neighbours)
                j = 0
                while j < len(seed_set):
                    q = seed_set[j]
                    
                    if labels[q] == 0:  # Change noise to border point
                        labels[q] = cid
                    elif labels[q] == -1:  # Unvisited point
                        labels[q] = cid
                        # Check if q is also a core point
                        q_neighbours = self.getNeighbours(X, q)
                        if len(q_neighbours) >= self.minPts:
                            # Add new neighbours to seed set
                            for n in q_neighbours:
                                if labels[n] == -1 or labels[n] == 0:
                                    if n not in seed_set:
                                        seed_set.append(n)
                    j += 1

        self.labels = labels
        self.n_clusters = cid  # Number of clusters found
        return labels 
    
    def getNeighbours(self, X: np.ndarray, pt_idx: int):
        """
        Find all neighbours of point at index pt_idx within eps radius
        """
        neighbours = []
        n_samples = X.shape[0]
        pt = X[pt_idx]
        for i in range(0, n_samples):
            if i != pt_idx:  # Don't include the point itself
                dist = self.euclidean_distance(X[i], pt)
                if dist <= self.eps:  # if other pt is in radius
                    neighbours.append(i)  # add to potential cluster
        return neighbours 










                










