import sys
import os
from optbinning import OptimalBinning

import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")


__all__ = [
    'CH_Binning'
]



class CH_Binning():
    def __init__(self,  max_n_bins:int=15):
        self.max_n_bins = max_n_bins


    def fit(self, x:list, y:list, metric:str='bins'):
        self.ch_model_ = 0
        for i in range(2, self.max_n_bins+1):
            model_ = OptimalBinning(dtype="numerical", solver="cp", max_n_bins=i)
            model_.fit(x, y)
            fitted_ = model_.transform(x, metric=metric)

            new_ch_model_ = CH_Binning.calinski_harabasz(y_pred=x, bins=fitted_)
            
            if (new_ch_model_ > self.ch_model_):
                self.ch_model_ = new_ch_model_
                self.n_bins_ = i
                self.model = model_

            self._copy_model_attributes()
            
        return self.model


    def fit_transform(self, x:list, y:list, metric:str='bins'):
        self.fit(x, y, metric)
        return self.transform(x, metric)


    def transform(self, x:list, metric:str='bins'):
        pred_ = self.model.transform(x, metric=metric)
        return pred_


    def _copy_model_attributes(self):
        for attr in dir(self.model):
            if not attr.startswith('_') and not callable(getattr(self.model, attr)):
                setattr(self, attr, getattr(self.model, attr))


    @staticmethod
    def map_to_alphabet_(self, lst):
        result = {num: chr(65 + index) for index, num in enumerate(lst)}
        self.df['rating'] = self.df['rating'].map(result).fillna('-')
        return result


    @staticmethod
    def calinski_harabasz(y_pred:list, bins:list):
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
            # print(f'(wss / (n - g)) == 0 | (wss = {wss}) (n = {n}) (g = {g}) | Optimum Might Have Been Achieved')
            return np.inf

        ch = (bss / (g - 1)) / (wss / (n - g))
        
        return float(round(ch,4))








if __name__ == '__main__':
    df = {
        'Grade': [0]*(95+309) + [1]*(187+224) + [2]*(549+299) + [3]*(1409+495) + [4]*(3743+690) + [5]*(4390+424) + [6]*(2008+94) + [7]*(593+8),
        'y_true': [0]*95+[1]*309 + [0]*187+[1]*224 + [0]*549+[1]*299 + [0]*1409+[1]*495 + [0]*3743+[1]*690 + [0]*4390+[1]*424 + [0]*2008+[1]*94 + [0]*593+[1]*8,
        'y_pred': [309/(95+309)]*(95+309) + [224/(187+224)]*(187+224) + [299/(549+299)]*(549+299) + [495/(1409+495)]*(1409+495) + [690/(3743+690)]*(3743+690) + [424/(4390+424)]*(4390+424) + [94/(2008+94)]*(2008+94) + [8/(593+8)]*(593+8)
    }
    df = pd.DataFrame(df)

    model = CH_Binning()
    model.fit(df['y_pred'], df['y_true'])
    print(model.transform(df['y_pred'],))