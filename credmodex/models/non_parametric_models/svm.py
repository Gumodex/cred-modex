from sklearn import svm


__all__ = [
    'SVM'
]


class SVM(svm.SVC):
    def __init__(self, C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0,
                 shrinking=True, tol=1e-3, cache_size=200, class_weight=None,
                 verbose=False, max_iter=-1, decision_function_shape='ovr',
                 break_ties=False, random_state=None):
        super().__init__(C=C, kernel=kernel, degree=degree, gamma=gamma,
                         coef0=coef0, shrinking=shrinking, probability=True,
                         tol=tol, cache_size=cache_size, class_weight=class_weight,
                         verbose=verbose, max_iter=max_iter,
                         decision_function_shape=decision_function_shape,
                         break_ties=break_ties, random_state=random_state)

    def predict(self, X):
        return self.predict_proba(X)[:,0]