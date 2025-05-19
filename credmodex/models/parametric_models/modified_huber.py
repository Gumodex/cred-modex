from sklearn.linear_model import SGDClassifier


__all__ = [
    'ModifiedHuber'
]


class ModifiedHuber(SGDClassifier):
    """
    Custom classifier that uses `loss='modified_huber'` and overrides
    the `predict` method to return class probabilities.

    This classifier behaves like a standard `SGDClassifier` with
    modified Huber loss, but `predict(X)` returns `predict_proba(X)`
    instead of class labels.
    """

    def __init__(self, penalty='l2', alpha=0.0001, fit_intercept=True,
                 max_iter=1000, tol=1e-3, random_state=None, 
                 learning_rate='optimal', eta0=0.0, power_t=0.5,
                 early_stopping=False, validation_fraction=0.1,
                 n_iter_no_change=5, class_weight=None, warm_start=False,
                 average=False):
        
        super().__init__(loss='modified_huber', penalty=penalty, alpha=alpha,
                         fit_intercept=fit_intercept, max_iter=max_iter, tol=tol,
                         random_state=random_state, learning_rate=learning_rate,
                         eta0=eta0, power_t=power_t, early_stopping=early_stopping,
                         validation_fraction=validation_fraction,
                         n_iter_no_change=n_iter_no_change,
                         class_weight=class_weight, warm_start=warm_start,
                         average=average)


    def predict(self, X):
        """
        Override `predict()` to return probabilities instead of class labels.
        For binary classification, this returns the probability of the positive class.
        """
        return super().predict_proba(X)[:,0]