from sklearn.naive_bayes import MultinomialNB


__all__ = [
    'Multinomial'
]


class Multinomial(MultinomialNB):
    def __init__(self, alpha=1.0, fit_prior=True, class_prior=None):
        super().__init__(alpha=alpha, fit_prior=fit_prior, class_prior=class_prior)


    def predict(self, X):
        return self.predict_proba(X)[:,0]