import itertools

import numpy as np
import pandas as pd
import scipy.stats
import statsmodels.stats



class Correlation:
    @staticmethod
    def covariance(x:list, y:list, **kwargs):
        value = np.cov(m=x, y=y, **kwargs)[0][1]
        return value


    @staticmethod
    def pearson(x:list, y:list, p_value:bool=False, **kwargs):
        value = scipy.stats.pearsonr(x=x, y=y, **kwargs)
        
        if (p_value == True):
            return value.statistic, value.pvalue
        
        return value.statistic
    

    @staticmethod
    def spearman(x:list, y:list, p_value:bool=False, **kwargs):
        value = scipy.stats.spearmanr(a=x, b=y, **kwargs)
        
        if (p_value == True):
            return value.statistic, value.pvalue
        
        return value.statistic
    

    @staticmethod
    def mahalanobis(x:list, y:np.matrix, **kwargs):
        x = np.array(x)
        y = np.array(y)
        if x.shape[0] != y.shape[1]:
            raise ValueError(f"Dimensão incompatível: x tem {x.shape[0]} elementos, y tem {y.shape[1]} variáveis.")
        
        mu = np.mean(y, axis=0)
        cov = np.cov(y, rowvar=False)
        cov_inv = np.linalg.pinv(cov)  # <- mais robusto que np.linalg.inv
        diff = x - mu
        value = np.sqrt(diff.T @ cov_inv @ diff)
        return value
    

    @staticmethod
    def variance_inflation_factor(df:pd.DataFrame, features:list):
        dff = df[features]
        dff = pd.get_dummies(dff, drop_first=True)
        dff = dff.apply(pd.to_numeric, errors='coerce')
        dff = dff.dropna()
        dff = dff.astype(float)

        vif_df = pd.DataFrame()
        vif_df['Variable'] = dff.columns
        vif_df['VIF'] = [
            round(statsmodels.stats.outliers_influence.variance_inflation_factor(dff.values, i),3)
            for i in range(dff.shape[1])
        ]

        anderson_conditions = [
            (vif_df['VIF'] < 1.8),
            (vif_df['VIF'] >= 1.8) & (vif_df['VIF'] < 5),
            (vif_df['VIF'] >= 5) & (vif_df['VIF'] < 10),
            (vif_df['VIF'] >= 10),
        ]
        anderson_values = ['No Multicol.', 'Moderate', 'Potential Multicol.', 'Strong Multicol.']
        vif_df['ANDERSON (2022)'] = np.select(anderson_conditions, anderson_values, '-')

        return vif_df
    

    @staticmethod
    def correlation(df:pd.DataFrame, features:list, numeric:bool=False):
        dff = df[features]
        if numeric:
            dff = pd.get_dummies(dff, drop_first=True)
            dff = dff.apply(pd.to_numeric, errors='coerce')
            dff = dff.astype(float)

        correlation_results = []
        for col1, col2 in itertools.combinations(dff.columns, 2):
            try:
                valid_data = dff[[col1, col2]].dropna()
                if valid_data.shape[0] > 1:
                    correlation = valid_data[col1].corr(valid_data[col2])
                else:
                    correlation = None
            except Exception:
                correlation = None
            
            correlation_results.append({
                'Column 1': col1,
                'Column 2': col2,
                'Correlation': correlation
            })

        correlation_df = pd.DataFrame(correlation_results)
        return correlation_df







if __name__ == '__main__':
    print(
        Correlation.mahalanobis([1, 2], [[2, 3], [5, 6], [4, 5]])

    )