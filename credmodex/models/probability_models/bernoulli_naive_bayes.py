from sklearn.naive_bayes import BernoulliNB


__all__ = [
    'Bernoulli'
]


class Bernoulli(BernoulliNB):
    def __init__(self, alpha=1.0, binarize=0.0, fit_prior=True, class_prior=None):
        super().__init__(alpha=alpha, binarize=binarize, fit_prior=fit_prior, class_prior=class_prior)


    def predict(self, X):
        return self.predict_proba(X)[:,0]
