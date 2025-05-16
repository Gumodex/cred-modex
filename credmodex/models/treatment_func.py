import pandas as pd
import sys
import os

sys.path.append(os.path.abspath('.'))
from credmodex.rating import CH_Binning

__all__ = [
    'TreatentFunc'
]

class TreatentFunc():
    def __init__(self, df:pd.DataFrame=None, target:str=None):
        self.df = df.copy(deep=True) 
        self.target = target
        self.forbidden_cols = ['split', self.target, 'score', 'rating']
        self.bins_map = {}


    def check_str_col_(self, col:list|str=None):
        if isinstance(col, str):
            col = [col]
        col = list(col)
        if col is None:
            raise ValueError("You must specify a column or list of columns.")
        col = self.df.select_dtypes(exclude=["number", "datetime"]).columns.tolist()
        col = [c for c in col if c not in self.forbidden_cols]
        return col


    def dummy_str_columns(self, col:list|str=None):
        col = self.check_str_col_(col)

        self.df[col] = self.df[col].fillna('Missing')
        self.df = pd.get_dummies(self.df, columns=col)

        return self.df
    

    def bin_str_columns(self, col:list|str=None, 
                               min_n_bins:int=2, max_n_bins:int=10):
        col = self.check_str_col_(col)

        if self.target is None:
            raise ValueError("You must specify a target.")

        for c in col:
            bins = CH_Binning(
                min_n_bins=min_n_bins, max_n_bins=max_n_bins,
                dtype='categorical'
            )
            self.df[c] = bins.fit_transform(
                x=self.df[c],
                y=self.df[self.target]
            )
            self.bins_map[c] = bins.bins_map

        return self.df
    

    def dummy_binned_str_columns(self, col:list|str=None, 
                                      min_n_bins:int=2, max_n_bins:int=10):
        col = self.check_str_col_(col)

        for c in col:
            self.bin_str_columns(
                col=c, min_n_bins=min_n_bins, max_n_bins=max_n_bins
            )
            self.dummy_str_columns(col=c)

        return self.df
    

    def sequential_str_columns(self, col:list|str=None, 
                                        min_n_bins:int=2, max_n_bins:int=10):
        col = self.check_str_col_(col)

        if self.target is None:
            raise ValueError("You must specify a target.")

        for c in col:
            bins = CH_Binning(
                min_n_bins=min_n_bins, max_n_bins=max_n_bins,
                dtype='categorical', transform_func='sequence'
            )
            self.df[c] = bins.fit_transform(
                x=self.df[c],
                y=self.df[self.target]
            )

            self.bins_map[c] = bins.bins_map

        return self.df