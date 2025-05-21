import pandas as pd
import numpy as np
from typing import Literal

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from . import BaseBinning


__all__ = [
    'DecisionTreeBinning'
]


class DecisionTreeBinning(BaseBinning):
    def __init__(self, min_n_bins: int = 2, max_n_bins: int = 15, n_bins: int = None,
                 dtype: Literal['numerical', 'categorical'] = 'numerical',
                 transform_func: Literal['alphabet', 'sequence', 'normalize'] = 'alphabet',
                 task_type: Literal['classification', 'regression'] = 'classification'):
        super().__init__(min_n_bins=min_n_bins, max_n_bins=max_n_bins, n_bins=n_bins,
                         dtype=dtype, transform_func=transform_func, solver='calinski-harabasz')
        self.task_type = task_type


    def _fit(self, x: list, y: list = None, n_bins: int = None):
        if self.dtype != 'numerical':
            raise TypeError("DecisionTreeBinning only supports numerical variables.")
        if y is None:
            raise ValueError("Target variable `y` must be provided for DecisionTreeBinning.")
        if n_bins is None:
            n_bins = self.n_bins

        x_array = np.array(x).reshape(-1, 1)

        # Choose appropriate decision tree model
        if self.task_type == 'classification':
            model = DecisionTreeClassifier(max_leaf_nodes=n_bins, min_samples_leaf=0.05)
        else:
            model = DecisionTreeRegressor(max_leaf_nodes=n_bins, min_samples_leaf=0.05)

        model.fit(x_array, y)
        thresholds = self._get_thresholds_from_tree(model)

        # Convert tree thresholds to bin edges
        self.bins = [float('-inf')] + sorted(thresholds) + [float('inf')]
        self.tree_model = model
        self.fitted = True
        return self


    def _get_thresholds_from_tree(self, model):
        # Extract split thresholds from decision tree
        tree = model.tree_
        thresholds = []
        for i in range(tree.node_count):
            if tree.children_left[i] != tree.children_right[i]:
                thresholds.append(tree.threshold[i])
        return thresholds


    def _transform(self, x):
        return pd.cut(x, bins=self.bins, include_lowest=True, labels=False)
