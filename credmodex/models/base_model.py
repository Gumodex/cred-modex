import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from datetime import timedelta
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import joblib
import inspect

from credmodex.rating import *



class BaseModel:
    """
    Base class for all models with advanced splitting functionality.
    """

    def __init__(self, model:type=None, treatment:type=None, df:pd.DataFrame=None, seed:int=42, doc:str=None,
                 features=None, target=None, predict_type:str=None, time_col:str=None, name:str=None):
        self.seed = seed
        np.random.seed(self.seed)
        self.model = model
        self.treatment = treatment
        self.df = df
        self.doc = doc
        self.features = features
        self.target = target
        self.time_col = time_col
        self.name = name

        self.predict_type = predict_type
        self.fit_predict()

        self.train_test_()

        if callable(self.model):
            self.model_code = inspect.getsource(self.model)
        else:
            self.model_code = None

        if callable(self.treatment):
            self.treatment_code = inspect.getsource(self.treatment)
        else:
            self.treatment_code = None

        if callable(self.doc):
            self.doc = inspect.getsource(self.doc)
        else:
            self.doc = None

        self.ratings = {}


    def train_test_(self):
        self.train = self.df[self.df['split'] == 'train']
        self.test = self.df[self.df['split'] == 'test']

        self.X_train = self.df[self.df['split'] == 'train'][self.features]
        self.X_test = self.df[self.df['split'] == 'test'][self.features]
        self.y_train = self.df[self.df['split'] == 'train'][self.target]
        self.y_test = self.df[self.df['split'] == 'test'][self.target]


    def fit_predict(self):
        self.df = self.treatment(self.df)
        self.train_test_()
        self.model = self.model.fit(self.X_train, self.y_train)

        predict_type = self.predict_type.lower().strip() if isinstance(self.predict_type, str) else None

        if predict_type and 'prob' in predict_type:
            if hasattr(self.model, 'predict_proba'):
                self.df['score'] = self.model.predict_proba(self.df[self.features])[:,0]
                self.df['score'] = self.df['score'].apply(lambda x: round(x,6))
                return 
            elif hasattr(self.model, 'decision_function'):
                scores = self.model.decision_function(self.df[self.features])
                self.df['score'] = scores
                self.df['score'] = self.df['score'].round(6)
                return
            else:
                raise AttributeError("Model doesn't support probability prediction.")
        
        elif (predict_type is None) or (predict_type == 'raw'):
            if hasattr(self.model, 'predict'):
                preds = self.model.predict(self.df[self.features])
                self.df['score'] = preds
                self.df['score'] = self.df['score'].round(6)
                return
            else:
                raise AttributeError("Model doesn't support raw predictions.")

        else:
            raise SystemError('No ``predict_type`` available')
    

    def add_rating(self, model:type=None, doc:str=None, type='score', optb_type:str='transform', name:str=None,
                   time_col:str=None):
        if model is None:
            model = CH_Binning(max_n_bins=15)
        if name is None:
            name = f'{model.__class__.__name__}_{len(self.ratings)+1}'
        if time_col is None:
            time_col = self.time_col
        
        rating = Rating(
            model=model, df=self.df, type=type, features=self.features, target=self.target, 
            optb_type=optb_type, doc=doc, time_col=time_col, name=name
            )
        self.ratings[name] = rating
        setattr(self, name, rating)
        
        # Set the self.rating to the last one defined
        self.rating = rating
        
        return model