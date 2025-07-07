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


    def fit(self, strategy):
        df = self.df.copy(deep=True)  # Use this as a working df to carry forward changes
        self.fitted_columns = df.columns.tolist()
        self.strategy = strategy

        for cols, funcs in strategy.items():
            if isinstance(cols, str):
                cols = (cols,)
            if not isinstance(funcs, (list, tuple)):
                funcs = [funcs]

            for col in cols:
                for func in funcs:
                    if isinstance(func, str):
                        if func not in self.methods:
                            raise ValueError(f"Method '{func}' not found in available methods: {list(self.methods.keys())}")
                        method = self.methods[func]
                        method_func = method()
                    else:
                        method_func = func

                    fitted_info = method_func(df=df, cols=col, target=self.target, fit=True)
                    self.pipeline.setdefault(col, []).append((method_func, fitted_info))

                    # Apply transform immediately after fit to update df
                    df = method_func(df=df, cols=col, target=self.target, fit=False, fitted_info=fitted_info)


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
            for method_func, fitted_info in steps:
                df = method_func(df=df, cols=col, target=self.target, fit=False, fitted_info=fitted_info)

        return df
    

    def fillna(self, value=0):
        def _fillna(df, cols, target, fit=True, fitted_info=None):
            cols = _check_cols(df, cols)
            if fit:
                return value
            else:
                for col in cols:
                    df[col] = df[col].fillna(fitted_info)
                return df
        _fillna.__method_name__ = 'fillna'
        return _fillna


    def exclude_columns(self):
        def _exclude_columns(df, cols, target, fit=True, fitted_info=None):
            cols_checked = _check_cols(df, cols)
            if fit:
                return cols_checked
            else:
                for col in fitted_info:
                    if col in df.columns:
                        del df[col]
                return df
        _exclude_columns.__method_name__ = 'exclude_columns'
        return _exclude_columns


    def include_columns(self):
        def _include_columns(df, cols, target, fit=True, fitted_info=None):
            cols_checked = _check_cols(df, cols)
            if fit:
                return cols_checked
            else:
                df = df.loc[:, df.columns.isin(fitted_info + TreatmentFunc().forbidden_cols)]
                return df
        _include_columns.__method_name__ = 'include_columns'
        return _include_columns


    def dummy_bin_str(self, min_n_bins=2, max_n_bins=10):
        def _dummy_bin_str(df, cols, target, fit=True, fitted_info=None):
            df = df.copy(deep=True)
            cols_checked = _check_str_cols(df, cols)

            if fit:
                bins = CH_Binning(
                    min_n_bins=min_n_bins, max_n_bins=max_n_bins,
                    dtype='categorical'
                )
                df[cols_checked[0]] = bins.fit(x=df[cols_checked[0]], y=df[target])
                return bins.bins_map
            else:
                for col in cols_checked:
                    df[col] = df[col].fillna('NaN')
                    flat_cats = list(set(cat for group in fitted_info.keys() for cat in eval(group)))
                    for cat in flat_cats:
                        df[f"{col}_{cat}"] = (df[col] == cat).astype(int)
                    df.drop(columns=col, inplace=True)
                return df
        _dummy_bin_str.__method_name__ = 'dummy_bin_str'
        return _dummy_bin_str


    def sequential_bin_str(self, min_n_bins=2, max_n_bins=10):
        def _sequential_bin_str(df, cols, target, fit=True, fitted_info=None):
            df = df.copy(deep=True)
            cols_checked = _check_str_cols(df, cols)

            if fit:
                binning = CH_Binning(
                    min_n_bins=min_n_bins, max_n_bins=max_n_bins,
                    dtype='categorical', transform_func='sequence'
                )
                df[cols_checked[0]] = binning.fit(x=df[cols_checked[0]], y=df[target])
                return binning
            else:
                for col in cols_checked:
                    df[col] = fitted_info.transform(df[col])
                return df
        _sequential_bin_str.__method_name__ = 'sequential_bin_str'
        return _sequential_bin_str


    def normalize_bin_str(self, min_n_bins=2, max_n_bins=10):
        def _normalize_bin_str(df, cols, target, fit=True, fitted_info=None):
            df = df.copy(deep=True)
            cols_checked = _check_str_cols(df, cols)

            if fit:
                binning = CH_Binning(
                    min_n_bins=min_n_bins, max_n_bins=max_n_bins,
                    dtype='categorical', transform_func='normalize'
                )
                df[cols_checked[0]] = binning.fit(x=df[cols_checked[0]], y=df[target])
                return binning
            else:
                for col in cols_checked:
                    df[col] = fitted_info.transform(df[col])
                return df
        _normalize_bin_str.__method_name__ = 'normalize_bin_str'
        return _normalize_bin_str


    def normalize_float(self, min_value=None, max_value=None, clip=True):
        def _normalize_float(df, cols, target, fit=True, fitted_info=None):
            cols_checked = _check_float_cols(df, cols)
            if fit:
                return {
                    'min': df[cols_checked[0]].min() if min_value is None else min_value,
                    'max': df[cols_checked[0]].max() if max_value is None else max_value,
                    'clip': clip
                }
            else:
                for col in cols_checked:
                    df[col] = (df[col] - fitted_info['min']) / (fitted_info['max'] - fitted_info['min'])
                    if fitted_info['clip']:
                        df[col] = df[col].clip(lower=fitted_info['min'], upper=fitted_info['max'])
                return df
        _normalize_float.__method_name__ = 'normalize_float'
        return _normalize_float


    def exclude_str_columns(self):
        def _exclude_str_columns(df, cols, target, fit=True, fitted_info=None):
            if fit:
                return True
            else:
                for col in cols:
                    if col in df.columns and col not in TreatmentFunc().forbidden_cols:
                        del df[col]
                return df
        _exclude_str_columns.__method_name__ = 'exclude_str_columns'
        return _exclude_str_columns


    def exclude_nan_rows(self):
        def _exclude_nan_rows(df, cols, target, fit=True, fitted_info=None):
            cols_checked = _check_cols(df, cols)
            if fit:
                return True
            else:
                for col in cols_checked:
                    df = df[df[col].notna()]
                return df
        _exclude_nan_rows.__method_name__ = 'exclude_nan_rows'
        return _exclude_nan_rows


    def datetime_parser(self, strftime=r'%Y-%m-%d'):
        def _datetime_parser(df, cols, target, fit=True, fitted_info=None):
            cols_checked = _check_datetime_cols(df, cols)
            if fit:
                return {'strftime': strftime}
            else:
                for col in cols_checked:
                    df[col] = df[col].dt.strftime(fitted_info['strftime'])
                return df
        _datetime_parser.__method_name__ = 'datetime_parser'
        return _datetime_parser


    def auto(self):
        def _auto(df, cols, target, fit=True, fitted_info=None):
            return None
        _auto.__method_name__ = 'auto'
        return _auto
    

    @property
    def pipeline_summary(self):
        summary = {}
        for col, steps in self.pipeline.items():
            summary[col] = []
            for func, fitted_info in steps:
                # Try to get a readable name
                if hasattr(func, '__method_name__'):
                    name = func.__method_name__
                elif hasattr(func, '__name__'):
                    name = func.__name__
                elif hasattr(func, '__class__'):
                    name = func.__class__.__name__
                else:
                    name = str(func)
                summary[col].append((name, fitted_info))
        return summary
