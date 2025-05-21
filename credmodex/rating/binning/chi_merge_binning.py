import pandas as pd

from sklearn.mixture import GaussianMixture

from . import BaseBinning


__all__ = [
    'GMMBinning'
]


class ChiMergeBinning(BaseBinning):
    def __init__(self, min_n_bins=2, max_n_bins=15, n_bins=None,
                 dtype='numerical', transform_func='alphabet', max_pvalue=0.05):
        super().__init__(min_n_bins, max_n_bins, n_bins,
                         dtype, transform_func, solver='n_bins')
        self.max_pvalue = max_pvalue


    def _fit(self, x, y=None, n_bins=None):
        if y is None:
            raise ValueError("Target variable `y` is required for ChiMerge binning.")
        if self.dtype != 'numerical':
            raise TypeError("ChiMergeBinning only supports numerical variables.")
        if n_bins is None:
            n_bins = self.n_bins

        df = pd.DataFrame({'x': x, 'y': y})
        df = df.sort_values('x')
        df['bin'] = pd.cut(df['x'], bins=n_bins, labels=False)

        # Simplified placeholder: we just use qcut for now
        # True ChiMerge would require iterative merging based on chi2 test
        self.bins = pd.qcut(x, q=n_bins, retbins=True, duplicates='drop')[1].tolist()
        self.bins[0] = float('-inf')
        self.bins[-1] = float('inf')
        self.fitted = True
        return self


    def _transform(self, x):
        return pd.cut(x, bins=self.bins, include_lowest=True, labels=False)
