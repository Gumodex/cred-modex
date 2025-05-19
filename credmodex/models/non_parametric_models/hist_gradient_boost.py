from sklearn.ensemble import HistGradientBoostingClassifier


__all__ = [
    'HistGradientBoost'
]

class HistGradientBoost(HistGradientBoostingClassifier):
    def __init__(self, loss='log_loss', learning_rate=0.1, max_iter=100,
                 max_leaf_nodes=31, max_depth=None, min_samples_leaf=20,
                 l2_regularization=0.0, max_bins=255, early_stopping='auto',
                 scoring='loss', validation_fraction=0.1, n_iter_no_change=10,
                 tol=1e-7, verbose=0, random_state=None, class_weight=None,
                 warm_start=False, monotonic_cst=None, interaction_cst=None):
        super().__init__(loss=loss, learning_rate=learning_rate, max_iter=max_iter,
                         max_leaf_nodes=max_leaf_nodes, max_depth=max_depth,
                         min_samples_leaf=min_samples_leaf, l2_regularization=l2_regularization,
                         max_bins=max_bins, early_stopping=early_stopping, scoring=scoring,
                         validation_fraction=validation_fraction,
                         n_iter_no_change=n_iter_no_change, tol=tol, verbose=verbose,
                         random_state=random_state, class_weight=class_weight,
                         warm_start=warm_start, monotonic_cst=monotonic_cst,
                         interaction_cst=interaction_cst)


    def predict(self, X):
        return self.predict_proba(X)[:,0]
