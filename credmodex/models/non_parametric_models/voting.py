from sklearn.ensemble import VotingClassifier


__all__ = [ 
    'Voting'
]


class Voting(VotingClassifier):
    def __init__(self, estimators, voting='soft', weights=None, n_jobs=None, flatten_transform=True, verbose=False):
        super().__init__(estimators=estimators, voting=voting, weights=weights, n_jobs=n_jobs,
                         flatten_transform=flatten_transform, verbose=verbose)


    def predict(self, X):
        return self.predict_proba(X)[:,0]