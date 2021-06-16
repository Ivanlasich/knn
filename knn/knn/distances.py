import numpy as np


def euclidean_distance(x, y):
    s1 = np.sum(x ** 2, axis=1)
    s2 = np.sum(y ** 2, axis=1)
    s = s1.reshape((x.shape[0], 1)) + s2 - 2 * x.dot(y.T)
    return np.sqrt(s)


def cosine_distance(x, y):
    s1 = np.sum(x ** 2, axis=1)
    s2 = np.sum(y ** 2, axis=1)
    return 1 - x.dot(y.T)/np.sqrt(s1.reshape((x.shape[0], 1)).dot(s2.reshape((1, y.shape[0]))))
