from xgboost import XGBClassifier

__all__ = [
    'XGBoost'
]

class XGBoost(XGBClassifier):
    def __init__(self,  objective='binary:logistic', learning_rate=0.1, n_estimators=100, max_depth=4, 
                 min_child_weight=1, gamma=0, subsample=1, colsample_bytree=1, reg_alpha=0, reg_lambda=1, 
                 base_score=0.5, random_state=None, eval_metric='logloss', **kwargs):
        super().__init__(objective=objective, learning_rate=learning_rate, n_estimators=n_estimators, 
                         max_depth=max_depth, min_child_weight=min_child_weight, gamma=gamma, subsample=subsample, 
                         colsample_bytree=colsample_bytree, reg_alpha=reg_alpha, reg_lambda=reg_lambda, 
                         base_score=base_score, random_state=random_state, eval_metric=eval_metric, **kwargs)

    def predict(self, X):
        return self.predict_proba(X)[:, 0]
