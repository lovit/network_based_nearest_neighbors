import numpy as np
from sklearn.metrics import pairwise_distances

class NetworkBasedNeighbors:
    def __init__(self, X=None, n_nearest_neighbors=5,
        n_random_neighbors=5, batch_size=500, verbose=True):

        self.n_nearest_neighbors = n_nearest_neighbors
        self.n_random_neighbors = n_random_neighbors
        self.batch_size = batch_size
        self.verbose = verbose
        self.buffer_factor = 3

        if X is not None:
            self.index(X)

    def index(self, X):
        n_data = X.shape[0]
        num_nn = self.n_nearest_neighbors
        num_rn = self.n_random_neighbors

        # set reference data
        self.X = X
        self.n_data = n_data

        if self.verbose:
            print('Indexing ...')

        # nearest neighbor indexing
        _, self.nn = most_closest_points(
            X, topk=num_nn+1, batch_size=self.batch_size,
            verbose=self.verbose)
        self.nn = self.nn[:,1:]

        # random neighbor indexing
        self.rn = np.random.randint(n_data, size=(n_data, num_rn))

        if self.verbose:
            print('Indexing was done')

    def search_neighbors(self, query, k=5, max_steps=10, converge=0.000001):
        dist, idxs, infos, process_time = self._search_neighbors_dev(
            query, kmax_steps, converge)
        return dist, idxs

    def _search_neighbors_dev(self, query, k=5, max_steps=10, converge=0.000001):
        buffer_size = self.buffer_factor * k
        dist, idxs = self._initialize(query, buffer_size)
        dist_sum = dist.sum()

        infos = []
        process_time = time.time()

        for step in range(max_steps):
            candi_idxs = np.unique(
                np.concatenate([idxs, _get_neighbors(idxs)])
            )

            candi_dist = pairwise_distances(
                query, index.X[candi_idxs]).reshape(-1)

            args = candi_dist.argsort()[:buffer_size]
            idxs_ = candi_idxs[args]
            dist_ = candi_dist[args]
            dist_sum_ = dist_.sum()

            diff = dist_sum - dist_sum_
            infos.append((step, dist_sum_, diff))

            if diff <= converge:
                break

            dist = dist_
            idxs = idxs_
            dist_sum = dist_sum_

        process_time = time.time() - process_time

        idxs_ = dist.argsort()[:k]
        idxs = idxs[idxs_]
        dist = dist[idxs_]

        return dist, idxs, infos, process_time

    def _initialize(self, query, k):
        idxs = np.random.randint(self.n_data, size=k)
        refx = self.X[idxs]
        dist = pairwise_distances(refx, query)
        return dist, idxs

    def _get_neighbors(self, base):
        neighbor_idxs = np.concatenate(
            [self.nn[base].reshape(-1),
             self.rn[base].reshape(-1)]
        )
        return neighbor_idxs