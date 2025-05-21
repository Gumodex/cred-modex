import pandas as pd
import numpy as np
from typing import Literal

from sklearn.isotonic import IsotonicRegression

from . import BaseBinning


__all__ = [
    'IsotonicBinning'
]


class IsotonicBinning(BaseBinning):
    def __init__(self, min_n_bins: int = 2, max_n_bins: int = 15, n_bins: int = None,
                 dtype: Literal['numerical', 'categorical'] = 'numerical',
                 transform_func: Literal['alphabet', 'sequence', 'normalize'] = 'alphabet'):
        super().__init__(min_n_bins=min_n_bins, max_n_bins=max_n_bins, n_bins=n_bins,
                         dtype=dtype, transform_func=transform_func, solver='calinski-harabasz')


    def _fit(self, x: list, y: list = None, n_bins: int = None):
        if self.dtype != 'numerical':
            raise TypeError("IsotonicBinning only supports numerical variables.")
        if y is None:
            raise ValueError("Target variable `y` must be provided for IsotonicBinning.")
        if n_bins is None:
            n_bins = self.n_bins

        x_array = np.array(x)
        y_array = np.array(y)

        # Fit isotonic regression to get a monotonic transformation of x
        self.isotonic_model = IsotonicRegression(out_of_bounds='clip')
        y_monotonic = self.isotonic_model.fit_transform(x_array, y_array)

        # Bin the transformed output into quantiles (or evenly spaced bins)
        self.bins = pd.qcut(y_monotonic, q=n_bins, retbins=True, duplicates='drop')[1].tolist()
        self.bins[0] = float('-inf')
        self.bins[-1] = float('inf')
        self.y_monotonic = y_monotonic
        self.fitted = True
        return self


    def _transform(self, x):
        x_array = np.array(x)
        y_monotonic = self.isotonic_model.transform(x_array)
        return pd.cut(y_monotonic, bins=self.bins, include_lowest=True, labels=False)
