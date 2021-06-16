import numpy as np

from knn.distances import euclidean_distance, cosine_distance


def get_best_ranks(ranks, top, axis=1, return_ranks=False):
    top_slice = (slice(None), ) * axis + (slice(-top, None, ), )
    inv_slice = (slice(None), ) * axis + (slice(None, None, -1), )
    indices = np.argpartition(ranks, -top, axis=axis)[top_slice]
    ranks_top = np.take_along_axis(ranks, indices, axis=axis)
    indices = np.take_along_axis(indices, ranks_top.argsort(axis=axis)[inv_slice], axis=axis)
    result = (indices,)
    if return_ranks:
        ranks_top = np.take_along_axis(ranks, indices, axis=axis)
        result = (-ranks_top,) + result
    return result if len(result) > 1 else result[0]


class NearestNeighborsFinder:
    def __init__(self, n_neighbors, metric="euclidean"):
        self.n_neighbors = n_neighbors

        if metric == "euclidean":
            self._metric_func = euclidean_distance
        elif metric == "cosine":
            self._metric_func = cosine_distance
        else:
            raise ValueError("Metric is not supported", metric)
        self.metric = metric

    def fit(self, X, y=None):
        self._X = X
        return self

    def kneighbors(self, X, return_distance=False):
        distances = self._metric_func(X, self._X)
        return get_best_ranks(-distances, self.n_neighbors, 1, return_distance)
