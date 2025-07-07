import sys
import os
import pandas as pd
import numpy as np

from functools import partial
import warnings

from typing import Union, Callable, Dict, Tuple, List

sys.path.append(os.path.abspath('.'))
from credmodex.rating import CH_Binning


__all__ = [
    'TreatentFunc'
]


def _check_cols(df: pd.DataFrame = None, cols: tuple = None):
    if df is None:
        raise ValueError("You must pass a valid DataFrame.")
    if cols is None:
        raise ValueError("You must specify a column or list of columns.")
    if not isinstance(cols, tuple):
        cols = tuple([cols] if isinstance(cols, str) else cols)
    for c in cols:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found in the DataFrame. Available columns: {df.columns.tolist()}")
    return cols


def _check_str_cols(df:pd.DataFrame=None, cols:tuple=None):
    cols = _check_cols(df, cols)
    cols = tuple([c for c in cols 
                  if (c in df.select_dtypes(exclude=["number", "datetime"]).columns.tolist()) 
                  and (c not in TreatmentFunc().forbidden_cols)])
    return cols


def _check_float_cols(df:pd.DataFrame=None, cols:tuple=None):
    cols = _check_cols(df, cols)
    cols = tuple([c for c in cols 
                 if (c in df.select_dtypes(include=["number"]).columns.tolist()) 
                 and (c not in TreatmentFunc().forbidden_cols)])
    return cols


def _check_datetime_cols(df:pd.DataFrame=None, cols:tuple=None):
    cols = _check_cols(df, cols)
    cols = tuple([c for c in cols 
                  if (c in df.select_dtypes(include=["datetime"]).columns.tolist()) 
                  and (c not in TreatmentFunc().forbidden_cols)])
    return cols


class TreatmentFunc:
    def __init__(self, df:pd.DataFrame=pd.DataFrame(), target:str=None):
        self.df = df.copy(deep=True)
        self.target = target
        self.forbidden_cols = ['split', 'score', 'rating', 'id', self.target]
        self.pipeline = {}

        self.methods = {
            'fillna': self.fillna,
            'exclude_columns': self.exclude_columns,
            'include_columns': self.include_columns,
            'dummy_bin_str': self.dummy_bin_str,
            'sequential_bin_str': self.sequential_bin_str,
            'normalize_bin_str': self.normalize_bin_str,
            'normalize_float': self.normalize_float,
            'exclude_str_columns': self.exclude_str_columns,
            'exclude_nan_rows': self.exclude_nan_rows,
            'datetime_parser': self.datetime_parser,
            'auto': self.auto,
        }


    def fit(self, strategy:Dict[Union[str, Tuple[str, ...]], Union[str, Callable, List[Union[str, Callable]]]]):
        for cols, funcs in strategy.items():
            if isinstance(cols, str):
                cols = (cols,)

            if not isinstance(funcs, (list, tuple)):
                funcs = [funcs]

            for col in cols:
                for func in funcs:
                    # Handle string case
                    if isinstance(func, str):
                        if func not in self.methods:
                            raise ValueError(f"Method '{func}' not found in available methods: {list(self.methods.keys())}")
                        method = self.methods[func]
                        method_name = func
                    else:
                        # If it's a partial or a direct function, derive name
                        method = func
                        method_name = func.func.__name__ if isinstance(func, partial) else func.__name__

                    fitted_info = method(df=self.df, cols=col, target=self.target, fit=True)
                    if col not in self.pipeline:
                        self.pipeline[col] = []

                    self.pipeline[col].append((method_name, fitted_info))


    def transform(self, df:pd.DataFrame=None):
        if df is None:
            df = self.df.copy(deep=True)
        else:
            df = df.copy(deep=True)

        if hasattr(self, "fitted_columns"):
            missing = [col for col in self.fitted_columns if col not in df.columns]
            for col in missing:
                df[col] = None
                warnings.warn(f"Columns {missing} are missing in transform() and were added with NaN values.")
            df = df.loc[:, self.fitted_columns]

            expected_dtypes = self.df[self.fitted_columns].dtypes.to_dict()
            actual_dtypes = df.dtypes.to_dict()
            type_mismatches = {col: (expected_dtypes[col], actual_dtypes.get(col)) 
                            for col in self.fitted_columns 
                            if col in actual_dtypes and expected_dtypes[col] != actual_dtypes[col]}
            if type_mismatches:
                warnings.warn(f"Column type mismatches found: {type_mismatches}")

        for col, steps in self.pipeline.items():
            for method_name, fitted_info in steps:
                if isinstance(method_name, str):
                    method = self.methods.get(method_name)
                else:
                    method = method_name  # If it's a callable

                df = method(df=df, cols=col, target=self.target, fit=False, fitted_info=fitted_info)

        return df
    

    @staticmethod
    def fillna(df:pd.DataFrame, cols:tuple, target:str, fit:bool=True, fitted_info:dict=None, value=0):
        cols = _check_cols(df, cols)
        if (fit == True):
            return {'fillna': value}
        else:
            for col in cols:
                df[col] = df[col].fillna(fitted_info['fillna'])
            return df


    @staticmethod
    def exclude_columns(df:pd.DataFrame, cols:tuple, target:str, fit:bool=True, fitted_info:dict=None):
        cols = _check_cols(df, cols)
        if (fit == True):
            return {'exclude': cols}
        else:
            for col in fitted_info['exclude']:
                if col in df.columns:
                    del df[col]
            return df
    

    @staticmethod
    def include_columns(df:pd.DataFrame, cols:tuple, target:str, fit:bool=True, fitted_info:dict=None):
        cols = _check_cols(df, cols)
        if (fit == True):
            return {'include': cols}
        else:
            df = df.loc[:, df.columns.isin(fitted_info['include'] + TreatmentFunc().forbidden_cols)]
        return df


    @staticmethod
    def _bin_str_columns(df:pd.DataFrame, cols:tuple, target:str, fit:bool=True, fitted_info:dict=None,
                         min_n_bins:int=2, max_n_bins:int=10):
        if (len(cols) > 1):
            raise(f'This is an internal method and holds only for one column, not f"{cols}"')
        
        if (fit == True):
            df = df.copy(deep=True)

            if (target is None):
                raise ValueError("You must specify a target.")

            bins = CH_Binning(
                min_n_bins=min_n_bins, max_n_bins=max_n_bins,
                dtype='categorical'
            )
            df[cols[0]] = bins.fit(x=df[cols[0]], y=df[target])
            bins = bins.bins_map

            return bins
        else:
            ...


    @staticmethod
    def dummy_bin_str(df:pd.DataFrame, cols:tuple, target:str, fit:bool=True, fitted_info:dict=None,
                      min_n_bins:int=2, max_n_bins:int=10):
        df = df.copy(deep=True)
        cols = _check_str_cols(df, cols)

        if (fit == True):
            fitted_info = TreatmentFunc._bin_str_columns(
                    df=df, cols=cols, target=target, fit=True,
                    min_n_bins=min_n_bins, max_n_bins=max_n_bins
                )

            return fitted_info

        else:
            for col in cols:
                df[col] = df[col].fillna('NaN')
                categories = list(fitted_info.keys())  # Fallback: unique category groups

                # Extract flat list of categories (from bin groups like "['a']", "['b', 'e']"...)
                flat_cats = []
                for group in categories:
                    cats = eval(group)  # CAUTION: group is like "['a', 'b']" (string)
                    flat_cats.extend(cats)

                flat_cats = list(set(flat_cats))
                for cat in flat_cats:
                    df[f"{col}_{cat}"] = (df[col] == cat).astype(int)

                df.drop(columns=col, inplace=True)

            return df


    @staticmethod
    def sequential_bin_str(df:pd.DataFrame, cols:tuple, target:str, fit:bool=True, fitted_info:dict=None,
                           min_n_bins:int=2, max_n_bins:int=10):
        df = df.copy(deep=True)
        cols = _check_str_cols(df, cols)

        if (fit == True):
            if (target is None):
                raise ValueError("You must specify a target.")

            fitted_info = CH_Binning(
                min_n_bins=min_n_bins, max_n_bins=max_n_bins,
                dtype='categorical', transform_func='sequence'
            )
            df[cols[0]] = fitted_info.fit(x=df[cols[0]], y=df[target])

            return fitted_info

        else:
            for col in cols:
                df[col] = fitted_info.transform(df[col])
                
            return df
       

    @staticmethod
    def normalize_bin_str(df:pd.DataFrame, cols:tuple, target:str, fit:bool=True, fitted_info:dict=None,
                          min_n_bins:int=2, max_n_bins:int=10):
        df = df.copy(deep=True)
        cols = _check_str_cols(df, cols)

        if (fit == True):
            if (target is None):
                raise ValueError("You must specify a target.")

            fitted_info = CH_Binning(
                min_n_bins=min_n_bins, max_n_bins=max_n_bins,
                dtype='categorical', transform_func='normalize'
            )
            df[cols[0]] = fitted_info.fit(x=df[cols[0]], y=df[target])

            return fitted_info

        else:
            for col in cols:
                df[col] = fitted_info.transform(df[col])
                
            return df
       

    @staticmethod
    def normalize_float(df:pd.DataFrame, cols:tuple, target:str, fit:bool=True, fitted_info:dict=None,
                        min_value:float=None, max_value:float=None, clip:bool=True):
        cols = _check_float_cols(df, cols)
        if (fit == True):
            for col in cols:
                min_value = df[col].min() if (min_value is None) else min_value
                max_value = df[col].max() if (max_value is None) else max_value
            return {'min': min_value, 'max': max_value, 'clip': clip}
        else:
            for col in cols:
                df[col] = (df[col] - fitted_info['min']) / (fitted_info['max'] - fitted_info['min'])
                if (fitted_info['clip'] == True):
                    df[col] = df[col].clip(lower=fitted_info['min'], upper=fitted_info['max'])
            return df


    @staticmethod
    def exclude_str_columns(df:pd.DataFrame, cols:tuple, target:str, fit:bool=True, fitted_info:dict=None):
        if (fit == True):
            return {'exclude_str_columns': True}
        else:
            for col in cols:
                if (col in df.columns) and (col not in TreatmentFunc().forbidden_cols):
                    del df[col]
            return df
    
    
    @staticmethod
    def exclude_nan_rows(df:pd.DataFrame, cols:tuple, target:str, fit:bool=True, fitted_info:dict=None):
        cols = _check_cols(cols)
        if (fit == True):
            return {'exclude_nan_rows': True}
        else:
            for col in cols:
                df = df[df[col].notna()]
            return df

    
    @staticmethod
    def datetime_parser(df:pd.DataFrame, cols:tuple, target:str, fit:bool=True, fitted_info:dict=None,
                        strftime:str=r'%Y-%m-%d'):
        cols = _check_datetime_cols(df, cols)
        if (fit == True):
            return {'strftime': strftime}
        else:
            for col in cols:
                df[col] = df[col].dt.strftime(fitted_info['strftime'])
            return df


    @staticmethod
    def auto(df:pd.DataFrame, cols:tuple, target:str, fit:bool=True, fitted_info:dict=None):
        return None