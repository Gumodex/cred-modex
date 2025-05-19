from sklearn.ensemble import AdaBoostClassifier


__all__ = [
    'AdaBoost'
]


class AdaBoost(AdaBoostClassifier):
    def __init__(self, estimator=None, n_estimators=50, learning_rate=1.0,
                 algorithm='SAMME.R', random_state=None):
        super().__init__(estimator=estimator, n_estimators=n_estimators,
                         learning_rate=learning_rate, algorithm=algorithm,
                         random_state=random_state)


    def predict(self, X):
        return self.predict_proba(X)[:,0]
