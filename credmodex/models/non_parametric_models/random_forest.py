from sklearn.ensemble import RandomForestClassifier


__all__ = [
    'RandomForest'
]


class RandomForest(RandomForestClassifier):
    def __init__(self, n_estimators=100, criterion='gini', max_depth=None,
                 min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                 max_features='sqrt', max_leaf_nodes=None, min_impurity_decrease=0.0,
                 bootstrap=True, oob_score=False, n_jobs=None, random_state=None,
                 verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0,
                 max_samples=None):
        super().__init__(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                         min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                         min_weight_fraction_leaf=min_weight_fraction_leaf,
                         max_features=max_features, max_leaf_nodes=max_leaf_nodes,
                         min_impurity_decrease=min_impurity_decrease, bootstrap=bootstrap,
                         oob_score=oob_score, n_jobs=n_jobs, random_state=random_state,
                         verbose=verbose, warm_start=warm_start, class_weight=class_weight,
                         ccp_alpha=ccp_alpha, max_samples=max_samples)


    def predict(self, X):
        return self.predict_proba(X)[:,0]