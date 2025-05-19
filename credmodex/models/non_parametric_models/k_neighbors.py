from sklearn.neighbors import KNeighborsClassifier


__all__ = [
    'KNeighbors'
]


class KNeighbors(KNeighborsClassifier):
    def __init__(self, n_neighbors=5, weights='uniform', algorithm='auto',
                 leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None):
        super().__init__(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm,
                         leaf_size=leaf_size, p=p, metric=metric, metric_params=metric_params,
                         n_jobs=n_jobs)


    def predict(self, X):
        return self.predict_proba(X)[:,0]
