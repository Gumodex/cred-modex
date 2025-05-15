import pandas as pd
import sys
import os

sys.path.append(os.path.abspath('.'))
from credmodex.rating import CH_Binning



class TreatentFunc():
    def __init__(self, df:pd.DataFrame=None, target:str=None):
        self.df = df.copy(deep=True) 
        self.target = target


    def dummy_columns(self, col:list|str=None):
        if col is None:
            raise ValueError("You must specify a column or list of columns.")

        if isinstance(col, str):
            col = [col]

        for c in col:
            if pd.api.types.is_numeric_dtype(self.df[c]):
                raise TypeError(f"Column '{c}' is numeric, so dummy encoding does not apply!")

        self.df[col] = self.df[col].fillna('Missing')
        self.df = pd.get_dummies(self.df, columns=col)

        return self.df
    

    def bin_categorical_column(self, col:str=None, 
                               min_n_bins:int=2, max_n_bins:int=10):
        if col is None:
            raise ValueError("You must specify a column.")
        if self.target is None:
            raise ValueError("You must specify a target.")

        if not isinstance(col, str):
            raise ValueError("You must specify a single column.")

        if pd.api.types.is_numeric_dtype(self.df[col]):
            raise TypeError(f"Column '{col}' is numeric, so dummy encoding does not apply!")

        bins = CH_Binning(
            min_n_bins=min_n_bins, max_n_bins=max_n_bins,
            dtype='categorical'
        )
        self.df[col] = bins.fit_transform(
            x=self.df[col],
            y=self.df[self.target]
        )
        self.bins_map = bins.bins_map

        return self.df
    

    def dummy_bin_categorical_columns(self, col:list|str=None, 
                                      min_n_bins:int=2, max_n_bins:int=10):
        if col is None:
            raise ValueError("You must specify a column or list of columns.")

        if isinstance(col, str):
            col = [col]

        for c in col:
            if pd.api.types.is_numeric_dtype(self.df[c]):
                raise TypeError(f"Column '{c}' is numeric, so dummy encoding does not apply!")
            
        for c in col:
            self.bin_categorical_column(
                col=c, min_n_bins=min_n_bins, max_n_bins=max_n_bins
            )
            self.dummy_columns(col=c)

        return self.df
    

    def categorical_to_numerical_column(self, col:str=None, 
                                        min_n_bins:int=2, max_n_bins:int=10):
        if col is None:
            raise ValueError("You must specify a column.")
        if self.target is None:
            raise ValueError("You must specify a target.")

        if not isinstance(col, str):
            raise ValueError("You must specify a single column.")

        if pd.api.types.is_numeric_dtype(self.df[col]):
            raise TypeError(f"Column '{col}' is numeric, so dummy encoding does not apply!")

        bins = CH_Binning(
            min_n_bins=min_n_bins, max_n_bins=max_n_bins,
            dtype='categorical', transform_func='sequence'
        )
        self.df[col] = bins.fit_transform(
            x=self.df[col],
            y=self.df[self.target]
        )
        self.bins_map = bins.bins_map

        return self.df