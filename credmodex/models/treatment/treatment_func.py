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


    def _check_col(self, col:list|str=None):
        if isinstance(col, str):
            col = [col]
        col = list(col)

        for c in col:
            if c not in self.df.columns:
                raise ValueError(f"Column '{c}' not found in the DataFrame.")
        if col is None:
            raise ValueError("You must specify a column or list of columns.")        
        return col


    def _check_str_col(self, col:list|str=None):
        col = self._check_col(col)
        col = [c for c in col 
               if (c in self.df.select_dtypes(exclude=["number", "datetime"]).columns.tolist()) 
               and (c not in self.forbidden_cols)]
        return col
    

    def _check_float_col(self, col:list|str=None):
        col = self._check_col(col)
        col = [c for c in col 
               if (c in self.df.select_dtypes(include=["number"]).columns.tolist()) 
               and (c not in self.forbidden_cols)]
        return col


    def dummy_str_columns(self, col:list|str=None):
        col = self._check_str_col(col)

        self.df[col] = self.df[col].fillna('Missing')
        self.df = pd.get_dummies(self.df, columns=col)

        return self.df[col]
    

    def _bin_str_columns(self, col:list|str=None, 
                        min_n_bins:int=2, max_n_bins:int=10):
        col = self._check_str_col(col)

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

        return self.df[col]
    

    def dummy_bin_str_columns(self, col:list|str=None, 
                              min_n_bins:int=2, max_n_bins:int=10):
        col = self._check_str_col(col)

        for c in col:
            self._bin_str_columns(
                col=c, min_n_bins=min_n_bins, max_n_bins=max_n_bins
            )
            self.dummy_str_columns(col=c)

        return self.df[col]
    

    def sequentialize_bin_str_columns(self, col:list|str=None, 
                                      min_n_bins:int=2, max_n_bins:int=10):
        col = self._check_str_col(col)

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

        return self.df[col]


    def normalize_bin_str_columns(self, col:list|str=None, 
                                  min_n_bins:int=2, max_n_bins:int=10):
        col = self._check_str_col(col)

        if self.target is None:
            raise ValueError("You must specify a target.")

        for c in col:
            bins = CH_Binning(
                min_n_bins=min_n_bins, max_n_bins=max_n_bins,
                dtype='categorical', transform_func='normalize'
            )
            self.df[c] = bins.fit_transform(
                x=self.df[c],
                y=self.df[self.target]
            )

            self.bins_map[c] = bins.bins_map

        return self.df[col]
    

    def fillna(self, col:list|str=None, value=0):
        col = self._check_col(col)
        for c in col:
            self.df[c] = self.df[c].fillna(value)

        return self.df[col]