import sys
import os
import warnings

import pandas as pd
import numpy as np
import sklearn
import sklearn.linear_model

sys.path.append(os.path.abspath('.'))
from credmodex.models.base_ import *

df = pd.read_csv(r'C:\Users\gustavo.filho\Documents\Python\Modules\Credit Risk\test\df.csv')





class Logistic(BaseModel_):
    def __init__(self, model:type=None):
        """
        Initializes the Logistic class with necessary attributes.

        :param df: The pandas DataFrame containing the data.
        :param features: The list of feature columns used for prediction.
        :param target: The target column for prediction.
        """
        super().__init__()
        if model:
            self.model = model


    def fit(self, X=None, y=None, model_kwargs=None):
        """
        Fits a logistic regression model and stores it along with predictions.

        :param model_kwargs: Additional keyword arguments to be passed to the model class.
        :param name: The name of the model to be stored, if not provided, defaults to a generated name.
        :return: The fitted model.
        """
        self.X = X

        if X is None or y is None:
            raise ValueError("Features and target must be present in the DataFrame")
        if model_kwargs is None:
            model_kwargs = {}

        X = X.dropna()
        y = y.dropna()

        self.model = sklearn.linear_model.LogisticRegression(**model_kwargs, solver='saga', max_iter=1000)
        self.model.fit(X, y)
        self.coef_()

        return self.model


    def predict(self, X):
        """
        Retrieves predictions for a specific model by name.

        :param name: The name of the model whose predictions you want to retrieve.
        :return: The predictions for the model.
        """
        return self.model.predict_proba(X)[:, 1]

    def coef_(self):
        self.equation = ''
        for coef, col in zip(self.model.coef_[0], self.X.columns):
            self.equation += f"{coef:.4f} \cdot {col} + "
        self.equation = self.equation.rstrip(" + ")
        self.equation = "$" + self.equation + "$"
        return self.model.coef_






















if __name__ == "__main__":
    ...