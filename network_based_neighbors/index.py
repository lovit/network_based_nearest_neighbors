import numpy as np
from sklearn.metrics import pairwise_distances

class NetworkBasedNeighbors:
    def __init__(self, X=None, n_nearest_neighbors=5,
        n_random_neighbors=5, batch_size=500, verbose=True):

        self.n_nearest_neighbors = n_nearest_neighbors
        self.n_random_neighbors = n_random_neighbors
        self.batch_size = batch_size
        self.verbose = verbose

        if X is not None:
            self.index(X)

    def index(self, X):
        n_data = X.shape[0]
        num_nn = self.n_nearest_neighbors
        num_rn = self.n_random_neighbors

        # set reference data
        self.X = X

        if self.verbose:
            print('Indexing ...')

        # nearest neighbor indexing
        _, self.nn = most_closest_points(
            X, topk=num_nn+1, batch_size=self.batch_size,
            verbose=self.verbose)

        # random neighbor indexing
        self.rn = np.random.randint(n_data, size=(n_data, num_rn))

        if self.verbose:
            print('Indexing was done')

    def search_neighbors(self, query, k=5):
        raise NotImplemented