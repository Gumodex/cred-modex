from sklearn.ensemble import BaggingClassifier


__all__ = [
    'Bagging'
]


class Bagging(BaggingClassifier):
    def __init__(self, estimator=None, n_estimators=10, max_samples=1.0,
                 max_features=1.0, bootstrap=True, bootstrap_features=False,
                 oob_score=False, warm_start=False, n_jobs=None, random_state=None,
                 verbose=0):
        super().__init__(estimator=estimator, n_estimators=n_estimators,
                         max_samples=max_samples, max_features=max_features,
                         bootstrap=bootstrap, bootstrap_features=bootstrap_features,
                         oob_score=oob_score, warm_start=warm_start, n_jobs=n_jobs,
                         random_state=random_state, verbose=verbose)


    def predict(self, X):
        return self.predict_proba(X)[:,0]
