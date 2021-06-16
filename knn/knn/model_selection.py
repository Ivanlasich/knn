from collections import defaultdict

import numpy as np

from sklearn.model_selection import KFold, BaseCrossValidator
from sklearn.metrics import accuracy_score

from knn.classification import KNNClassifier, BatchedKNNClassifier


def knn_cross_val_score(X, y, k_list, scoring, cv=None, metric='euclidean', weights='uniform', algorithm='brute', batch_size = 100):
    y = np.asarray(y)
    if scoring == "accuracy":
        scorer = accuracy_score
    else:
        raise ValueError("Unknown scoring metric", scoring)

    if cv is None:
        cv = KFold(n_splits=5)
    elif not isinstance(cv, BaseCrossValidator):
        raise TypeError("cv should be BaseCrossValidator instance", type(cv))

    answer = {}
    for k in k_list:
        answer[k] = []

    for train_index, test_index in cv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = BatchedKNNClassifier(n_neighbors=max(k_list), algorithm=algorithm, metric=metric, weights=weights)
        clf.set_batch_size(batch_size)
        clf.fit(X_train, y_train)
        distances, indices = clf.kneighbors(X_test, return_distance=True)
        for k in k_list:
            clf._finder.n_neighbors = k
            predict = clf._predict_precomputed(indices[:, :k], distances[:, :k])
            acc = scorer(y_test, predict)
            answer[k].append(acc)

    return answer
