import numpy as np

from sklearn.mixture import GaussianMixture

from . import BaseBinning


__all__ = [
    'GMMBinning'
]


class GaussianMixBinning(BaseBinning):
    def __init__(self, min_n_bins=2, max_n_bins=15, n_bins=None,
                 dtype='numerical', transform_func='alphabet'):
        super().__init__(min_n_bins, max_n_bins, n_bins,
                         dtype, transform_func, solver='calinski-harabasz')


    def _fit(self, x, y=None, n_bins=None):
        if self.dtype != 'numerical':
            raise TypeError("GMMBinning only supports numerical variables.")
        if n_bins is None:
            n_bins = self.n_bins

        x_array = np.array(x).reshape(-1, 1)
        self.gmm = GaussianMixture(n_components=n_bins, random_state=42)
        self.gmm.fit(x_array)
        self.fitted = True
        return self


    def _transform(self, x):
        x_array = np.array(x).reshape(-1, 1)
        return self.gmm.predict(x_array)
