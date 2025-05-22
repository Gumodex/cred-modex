import warnings
import numpy as np
import pandas as pd

from typing import Literal

from abc import ABC, abstractmethod


__all__ = [
    'BaseBinning'
]


class BaseBinning(ABC):
    def __init__(self, min_n_bins:int=2, max_n_bins:int=15, n_bins:int=None,
                 dtype:Literal['numerical','categorical']='numerical',
                 transform_func:Literal['alphabet','sequence','normalize']='alphabet',
                 solver:Literal['calinski-harabasz']='calinski-harabasz'):
        super().__init__()
        self.max_n_bins = max_n_bins
        self.min_n_bins = min_n_bins
        self.n_bins = n_bins
        self.dtype = dtype
        self.transform_func = transform_func
        self.solver = solver
        self.fitted = False

        if (n_bins is not None):
            self.min_n_bins = self.n_bins
            self.max_n_bins = self.n_bins
            self.solver = 'n_bins'
        
        self.model = None


    def fit(self, x:list, y:list=None):
        if (self.solver == 'calinski-harabasz'):
            return self._fit_ch(x, y)
        else:
            try: 
                return self._fit(x, y, n_bins=self.n_bins)
            except:
                raise ValueError('You must specify a ``n_bins`` or a solver!')


    def _fit_ch(self, x:list, y:list=None):
        self.ch_model_ = -1
        self.ch_model_dict_ = {}
        best_model = None  # Track best model

        for i in range(self.min_n_bins, self.max_n_bins + 1):
            try:
                model_ = self._fit(x, y, n_bins=i)
            except Exception as e:
                continue

            try:
                fitted_ = model_._transform(x)
            except Exception:
                continue

            try:
                if self.dtype == 'categorical':
                    df = pd.DataFrame({'bin': fitted_, 'target': y})
                    bins_map = df.groupby('bin')['target'].mean().to_dict()
                    y_pred = [bins_map[b] for b in fitted_]
                else:
                    y_pred = x
            except:
                continue

            new_ch_model_ = BaseBinning.calinski_harabasz(y_pred=y_pred, bins=fitted_)
            self.ch_model_dict_[i] = new_ch_model_

            if new_ch_model_ > self.ch_model_ and not np.isinf(new_ch_model_):
                self.ch_model_ = new_ch_model_
                self.n_bins_ = i
                best_model = model_

        if best_model is None:
            print("⚠️ No valid model selected. CH scores:", self.ch_model_dict_)
            raise TypeError("No optimum binning was found.")

        self.model = best_model
        self.fitted = True
        return self.model


    @abstractmethod
    def _transform(self, x:list):
        return self.model._transform(x)


    def transform(self, x:list):
        if not self.fitted or self.model is None:
            raise ValueError("Model not fitted. Call `fit()` first.")
        return self.model._transform(x)


    def fit_transform(self, x:list, y:list=None):
        self.fit(x, y)
        return self.model.transform(x)

    
    @staticmethod
    def calinski_harabasz(y_pred:list, bins:list):
        """
        Compute the Calinski-Harabasz score to evaluate the separation between binned groups.

        This score measures how well predicted probabilities are clustered by a given binning scheme.
        It is commonly used in clustering and binning evaluations to assess group distinctiveness.

        Parameters
        ----------
        y_pred : list
            Predicted probabilities or scores (continuous values).
        bins : list
            Corresponding bin labels for each prediction (can be strings or numeric identifiers).

        Returns
        -------
        float
            Calinski-Harabasz score, rounded to 4 decimal places.
            Returns `np.inf` if the denominator is zero (suggesting perfect separation).
            Returns `0` if the result is not a number (`NaN`).

        Notes
        -----
        The score is calculated as:

            CH = (BSS / (g - 1)) / (WSS / (n - g))

        where:
        - BSS = between-group sum of squares
        - WSS = within-group sum of squares
        - g = number of bins
        - n = total number of observations

        A higher score indicates better separation between the groups.
        """
        df = {
            "y_pred": y_pred,
            "bins": bins
        }
        df = pd.DataFrame(df)
        
        overall_mean = df['y_pred'].mean()
        n = len(df)
        g = df['bins'].nunique()

        bss = (
            df.groupby('bins')['y_pred']
            .apply(lambda x: len(x) * (x.mean() - overall_mean) ** 2)
            .sum()
        )

        wss = (
            df.groupby('bins')['y_pred']
            .apply(lambda x: ((x - x.mean()) ** 2).sum())
            .sum()
        )

        if ((wss / (n - g)) == 0):
            return np.inf

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            ch = (bss / (g - 1)) / (wss / (n - g))

        if np.isnan(ch):
            return 0
        
        return float(round(ch,4))
    

    @staticmethod
    def _map_to_alphabet(df, lst):
        result = {num: chr(65 + index) for index, num in enumerate(lst)}
        df['rating'] = df['rating'].map(result).fillna('-')
        return result
