from sklearn.ensemble import StackingClassifier


__all__ = [ 
    'Stacking'
]


class Stacking(StackingClassifier):
    def __init__(self, estimators, final_estimator=None, cv=None, stack_method='auto',
                 passthrough=False, n_jobs=None, verbose=0):
        super().__init__(estimators=estimators, final_estimator=final_estimator, cv=cv,
                         stack_method=stack_method, passthrough=passthrough, n_jobs=n_jobs,
                         verbose=verbose)


    def predict(self, X):
        return self.predict_proba(X)[:,0]