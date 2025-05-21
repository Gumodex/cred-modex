import pandas as pd

from . import BaseBinning


__all__ = [
    'HandBinning'
]


class HandBinning(BaseBinning):
    def __init__(self, bins: list, dtype='numerical', transform_func='alphabet'):
        super().__init__(n_bins=len(bins) - 1, dtype=dtype,
                         transform_func=transform_func, solver='n_bins')
        self.user_bins = sorted(set(bins))

    def _fit(self, x, y=None, n_bins=None):
        if self.dtype != 'numerical':
            raise TypeError("HandBinning only supports numerical variables.")
        if len(self.user_bins) < 2:
            raise ValueError("You must provide at least two bin edges.")

        # Only expand if needed
        self.bins = self.user_bins.copy()
        if self.bins[0] > min(x):
            self.bins[0] = float('-inf')
        if self.bins[-1] < max(x):
            self.bins[-1] = float('inf')

        self.fitted = True
        return self

    def _transform(self, x):
        return pd.cut(x, bins=self.bins, include_lowest=True, labels=False)
