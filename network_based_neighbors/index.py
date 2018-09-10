import numpy as np
from sklearn.metrics import pairwise_distances

class NetworkBasedNeighbors:
    def __init__(self, X=None, n_nearest_neighbors=5,
        n_random_neighbors=5, batch_size=500, verbose=True):

        self.n_nearest_neighbors = n_nearest_neighbors
        self.n_random_neighbors = n_random_neighbors
        self.batch_size = batch_size
        self.vebose = verbose

        if X is not None:
            self.index(X)

    def index(self, X):
        raise NotImplemented

    def search_neighbors(self, query, k=5):
        raise NotImplemented