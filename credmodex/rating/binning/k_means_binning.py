import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from typing import Literal
from . import BaseBinning


__all__ = [
    'KMeansBinning'
]


class KMeansBinning(BaseBinning):
    def __init__(self, min_n_bins:int=2, max_n_bins:int=15, n_bins:int=None,
                 dtype:Literal['numerical', 'categorical']='numerical',
                 transform_func:Literal['alphabet', 'sequence', 'normalize']='alphabet'):
        super().__init__(min_n_bins=min_n_bins, max_n_bins=max_n_bins, n_bins=n_bins,
                         dtype=dtype, transform_func=transform_func, solver='calinski-harabasz')
    
    def _fit(self, x:list, y:list=None, n_bins:int=None):
        if self.dtype != 'numerical':
            raise TypeError("KMeansBinning only supports numerical variables.")

        if n_bins is None:
            n_bins = self.n_bins

        x_array = np.array(x).reshape(-1, 1)
        self.kmeans = KMeans(n_clusters=n_bins, random_state=42, n_init=10)
        self.kmeans.fit(x_array)

        # Get cluster centers and sort them to determine bin edges
        centers = sorted(self.kmeans.cluster_centers_.flatten())
        midpoints = [(centers[i] + centers[i+1]) / 2 for i in range(len(centers) - 1)]

        # Extend to cover full range
        self.bins = [float('-inf')] + midpoints + [float('inf')]
        self.fitted = True
        return self


    def _transform(self, x):
        return pd.cut(x, bins=self.bins, include_lowest=True, labels=False)
