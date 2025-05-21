import pandas as pd
import numpy as np
from typing import Literal

from . import BaseBinning


__all__ = [
    'QuantileBinning'
]


class QuantileBinning(BaseBinning):
    def __init__(self, min_n_bins:int=2, max_n_bins:int=15, n_bins:int=None,
                 dtype:Literal['numerical','categorical']='numerical',
                 transform_func:Literal['alphabet','sequence','normalize']='alphabet'):
        super().__init__(min_n_bins=min_n_bins, max_n_bins=max_n_bins, n_bins=n_bins,
                         dtype=dtype, transform_func=transform_func, solver='calinski-harabasz')


    def _fit(self, x:list, y:list=None, n_bins:int=None):
        if self.dtype == 'categorical':
            raise TypeError("QuantileBinning does not support categorical variables.")
        if (n_bins is None):
            n_bins = self.n_bins
        
        self.bins = pd.qcut(x, q=n_bins, retbins=True, duplicates='drop')[1].tolist()
        self.bins[0] = -np.inf
        self.bins[-1] = np.inf
        self.fitted = True
        return self


    def _transform(self, x):
        return pd.cut(x, bins=self.bins, include_lowest=True, labels=False)
