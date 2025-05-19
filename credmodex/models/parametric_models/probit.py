from statsmodels.discrete import discrete_model


__all__ = [
    'Probit'
]


class Probit(discrete_model.Probit):
    def __init__(self, max_iter=1000, solver='bfgs'):
        self.max_iter = max_iter
        self.solver = solver

    
    def fit(self, X, y, **kwargs):
        self.model = discrete_model.Probit(
            endog=y, exog=X, **kwargs
        )
        self.X = X
        self.y = y
        self.model = self.model.fit(maxiter=self.max_iter, method=self.solver)
        return self.model
    
    
    def predict(self, X):
        return [1-x for x in self.model.predict(X).to_list()]