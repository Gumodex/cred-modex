from optbinning import OptimalBinning
from typing import Literal

import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

from . import BaseBinning


__all__ = [
    'CC'
]


class CC(BaseBinning):
    def __init__(self, min_n_bins: int = 2, max_n_bins: int = 15, n_bins: int = None,
                 dtype: Literal['numerical', 'categorical'] = 'numerical',
                 transform_func: Literal['alphabet', 'sequence', 'normalize'] = 'alphabet'):
        super().__init__(min_n_bins=min_n_bins, max_n_bins=max_n_bins, n_bins=n_bins,
                         dtype=dtype, transform_func=transform_func, solver='calinski-harabasz')


    def _fit(self, x: list, y: list, n_bins: int = None):
        if y is None:
            raise ValueError("Target variable `y` must be provided for OptimalBinning.")
        if n_bins is None:
            n_bins = self.n_bins

        self.model = OptimalBinning(
            dtype=self.dtype,
            solver="cp",
            min_n_bins=n_bins,
            max_n_bins=n_bins
        )
        self.model.fit(x, y)
        self.fitted = True
        return self


    def _transform(self, x: list):
        if not self.fitted:
            raise ValueError("You must call `fit` before `transform`.")
        return self.model.transform(x, metric="bins")


    # def _fit(self, x:list, y:list, n_bins:int=None):
    #     if y is None:
    #         raise ValueError("Target variable `y` must be provided for DecisionTreeBinning.")
    #     if n_bins is None:
    #         n_bins = self.n_bins

    #     self.opt_model_ = OptimalBinning(dtype=self.dtype, solver="cp", min_n_bins=n_bins, max_n_bins=n_bins)
    #     self.opt_model_.fit(x, y)

    #     return self


    # def _transform(self, x:list):
    #     if (self.dtype == 'numerical'):
    #         pred_ = self.model.transform(x, metric='bins')
    #         return pred_
    #     if (self.dtype == 'categorical'):
    #         pred_ = self.model.transform(x, metric='bins')
    #         if (self.transform_func == 'alphabet'):
    #             pred_ = self._convert_categorical(bins=self.bins_map, list_=pred_)
    #         elif (self.transform_func == 'sequence'):
    #             pred_ = self._convert_categorical_to_numerical(bins=self.bins_map, list_=pred_)
    #         elif (self.transform_func == 'normalize'):
    #             pred_ = self._convert_categorical_to_normal(bins=self.bins_map, list_=pred_)
    #         return pred_


    def _copy_model_attributes(self):
        for attr in dir(self.model):
            if not attr.startswith('_') and not callable(getattr(self.model, attr)):
                setattr(self, attr, getattr(self.model, attr))


    def _map_to_alphabet(self, lst):
        result = {num: chr(65 + index) for index, num in enumerate(lst)}
        self.df['rating'] = self.df['rating'].map(result).fillna('-')
        return result


    def _convert_categorical(self, bins:dict, list_:list):
        bins = dict(sorted(bins.items(), key=lambda item: item[1]))
        letter = 0
        for key in list(bins.keys()):
            if key == 'Missing':
                bins[key] = 'None'
            else:
                bins[key] = chr(65 + letter)
                letter += 1

        self.bins_map = bins
        return [bins[label] for label in list_]


    def _convert_categorical_to_numerical(self, bins:dict, list_:list):
        bins = dict(sorted(bins.items(), key=lambda item: item[1]))
        weight = [round(1-(i/len(bins.keys())),6) for i in range(len(bins.keys()))]
        for key, value in zip(bins.keys(), weight):
            bins[key] = value

        self.bins_map = bins
        return [bins[label] for label in list_]


    def _convert_categorical_to_normal(self, bins:dict, list_:list):
        bins = dict(sorted(bins.items(), key=lambda item: item[1]))
        values = list(bins.values())
        _min = min(values)
        _max = max(values)
        for key, value in bins.items():
            bins[key] = round(abs((value - _max) / (_min - _max)),6)

        self.bins_map = bins
        return [bins[label] for label in list_]
