from sklearn.naive_bayes import GaussianNB


__all__ = [
    'Gaussian'
]


class Gaussian(GaussianNB):
    def __init__(self, priors=None, var_smoothing=1e-9):
        super().__init__(priors=priors, var_smoothing=var_smoothing)


    def predict(self, X):
        return self.predict_proba(X)[:,0]