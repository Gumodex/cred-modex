import sys
import os
import warnings

import pandas as pd
import numpy as np

sys.path.append(os.path.abspath('.'))
from credmodex.discriminancy.discrete import *

df = pd.read_csv(r'C:\Users\gustavo.filho\Documents\Python\Modules\Credit Risk\test\df.csv')





class CredLab:
    def __init__(self, df:pd.DataFrame=None, target:str=None, features:list[str]=None):

        if df is None:
            raise ValueError("DataFrame cannot be None")
        self.raw_df = df
        if (not isinstance(target, str)) or (not isinstance(features, list)):
            raise ValueError("target must be a string and features must be a list of strings")
        # The self.df contains only the columns target + features
        features = [f for f in features if f in df.columns and f != target]
        self.df = df[features + [target]] if features and target else None
        if self.df is None:
            raise ValueError("Both target and [features] must be provided.")
        
        self.target = target
        self.features = features


    def add_model(self):
        return None


    def eval_metric(self):
        return None
    
    
    def eval_plot(self):
        return None
    

    def eval_discriminancy(self, method:str='iv', conditions:list=[], plot:bool=False,):
        df = self.df.copy()
        for condition in conditions:
            df = df.query(condition)
        if df.empty:
            print("DataFrame is empty after applying conditions!")
            return None

        if method == 'iv':
            return IV_Discriminant(df, self.target, self.features)
        if method == 'ks':
            return KS_Discriminant(df, self.target, self.features)









if __name__ == "__main__":
    project = CredLab(df, target='over', features=df.columns.to_list())
    print(project.eval_discriminancy('ks'))