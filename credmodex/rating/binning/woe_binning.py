import pandas as pd
import numpy as np
from typing import Literal

from . import BaseBinning


__all__ = [
    'WoeBinning'
]


class WoeBinning(BaseBinning):
    def __init__(self, min_n_bins=2, max_n_bins=15, n_bins=None,
                 dtype='numerical', transform_func='alphabet'):
        super().__init__(min_n_bins, max_n_bins, n_bins,
                         dtype, transform_func, solver='n_bins')

    def _fit(self, x, y=None, n_bins=None):
        if y is None:
            raise ValueError("Target variable `y` is required for WOE binning.")
        if self.dtype != 'numerical':
            raise TypeError("WOE binning only supports numerical variables.")
        if n_bins is None:
            n_bins = self.n_bins

        df = pd.DataFrame({'x': x, 'y': y})
        df['bin'] = pd.qcut(df['x'], q=n_bins, duplicates='drop')

        # Calculate WOE and IV
        grouped = df.groupby('bin')
        total_pos = (df['y'] == 1).sum()
        total_neg = (df['y'] == 0).sum()

        woe_dict = {}
        iv = 0
        for name, group in grouped:
            pos = (group['y'] == 1).sum()
            neg = (group['y'] == 0).sum()
            rate_pos = pos / total_pos if total_pos else 1
            rate_neg = neg / total_neg if total_neg else 1
            woe = np.log(rate_pos / rate_neg) if rate_neg and rate_pos else 0
            woe_dict[name] = woe
            iv += (rate_pos - rate_neg) * woe

        self.woe_dict = woe_dict
        self.iv = iv
        self.bins = pd.qcut(x, q=n_bins, retbins=True, duplicates='drop')[1].tolist()
        self.bins[0] = float('-inf')
        self.bins[-1] = float('inf')
        self.fitted = True
        return self

    def _transform(self, x):
        binned = pd.cut(x, bins=self.bins, include_lowest=True)
        return binned.map(self.woe_dict).fillna(0)