import itertools
import sys
import os

import numpy as np
import pandas as pd
import scipy.stats
from statsmodels.stats import outliers_influence

import plotly.graph_objects as go
sys.path.append(os.path.abspath('.'))
from credmodex.utils.design import *


__all__ = [
    'Correlation'
]


class Correlation:
    """
    A comprehensive set of static methods for evaluating statistical relationships 
    and multicollinearity diagnostics among variables.

    The `Correlation` class includes tools for:
    - Measuring linear and monotonic associations (Pearson, Spearman)
    - Computing covariance and Mahalanobis distance
    - Assessing multicollinearity via Variance Inflation Factor (VIF)
    - Generating pairwise correlation matrices between features

    This utility is particularly useful for exploratory data analysis (EDA),
    feature selection, and multivariate diagnostics in predictive modeling.

    All methods are implemented as `@staticmethod` to allow flexible,
    modular use without instantiating the class.

    Examples
    --------
    >>> Correlation.pearson([1, 2, 3], [4, 5, 6])
    1.0
    --------
    >>> Correlation.variance_inflation_factor(df, ['age', 'income', 'credit_score'])
       Variable         VIF     ANDERSON (2022)
    0  age             1.21     No Multicol.
    1  income          3.45     Moderate
    2  credit_score    9.13     Potential Multicol.
    --------
    >>> Correlation.correlation(df, ['age', 'income'], numeric=True)
       Column 1   Column 2   Correlation
    0  age        income            0.84
    --------
    >>> Correlation.mahalanobis([1, 2], np.array([[0, 0], [1, 1], [2, 2]]))
    1.41
    """

    @staticmethod
    def covariance(x:list, y:list, **kwargs):
        """
        Compute the covariance between two variables.

        Covariance measures the joint variability of two variables.
        A positive value indicates that the variables increase together,
        while a negative value suggests an inverse relationship.

        Parameters
        ----------
        x : list
            First input variable.
        y : list
            Second input variable.
        **kwargs : dict, optional
            Additional arguments passed to `np.cov`.

        Returns
        -------
        float
            Covariance value between x and y.
        """
        value = np.cov(m=x, y=y, **kwargs)[0][1]
        return value


    @staticmethod
    def pearson(x:list, y:list, info:bool=False, **kwargs):
        """
        Compute the Pearson correlation coefficient between two variables.

        Pearson correlation measures the linear relationship between two variables.
        It returns a value between -1 and 1, where 1 indicates perfect positive
        correlation, -1 perfect negative correlation, and 0 no linear relationship.

        Parameters
        ----------
        x : list
            First input variable.
        y : list
            Second input variable.
        info : bool, optional
            If True, also returns the p-value of the correlation.
        **kwargs : dict, optional
            Additional arguments passed to `scipy.stats.pearsonr`.

        Returns
        -------
        float or tuple
            Correlation coefficient, or a tuple of (coefficient, p-value) if `info` is True.
        """
        value = scipy.stats.pearsonr(x=x, y=y, **kwargs)
        
        if (info == True):
            return value.statistic, value.pvalue
        
        return value.statistic
    

    @staticmethod
    def spearman(x:list, y:list, info:bool=False, **kwargs):
        """
        Compute the Spearman rank correlation coefficient between two variables.

        Spearman correlation assesses how well the relationship between two variables
        can be described using a monotonic function. It is based on ranked values
        and is less sensitive to outliers than Pearson correlation.

        Parameters
        ----------
        x : list
            First input variable.
        y : list
            Second input variable.
        info : bool, optional
            If True, also returns the p-value of the correlation.
        **kwargs : dict, optional
            Additional arguments passed to `scipy.stats.spearmanr`.

        Returns
        -------
        float or tuple
            Spearman correlation coefficient, or a tuple of (coefficient, p-value) if `info` is True.
        """
        value = scipy.stats.spearmanr(a=x, b=y, **kwargs)
        
        if (info == True):
            return value.statistic, value.pvalue
        
        return value.statistic
    

    @staticmethod
    def mahalanobis(x:list, y:np.matrix, **kwargs):
        """
        Compute the Mahalanobis distance between a vector and a multivariate dataset.

        Mahalanobis distance measures how many standard deviations away a point is
        from the mean of a distribution, considering the covariance structure.
        It is useful for multivariate outlier detection or similarity measurement.

        Parameters
        ----------
        x : list
            Observation vector to compare (length must match number of columns in y).
        y : np.ndarray
            Multivariate dataset, each row is an observation, columns are variables.
        **kwargs : dict, optional
            Additional arguments for customization (currently unused).

        Returns
        -------
        float
            Mahalanobis distance between x and the distribution defined by y.

        Raises
        ------
        ValueError
            If dimensions of x and y are incompatible.
        """
        x = np.array(x)
        y = np.array(y)
        if x.shape[0] != y.shape[1]:
            raise ValueError(f"Dimensão incompatível: x tem {x.shape[0]} elementos, y tem {y.shape[1]} variáveis.")
        
        mu = np.mean(y, axis=0)
        cov = np.cov(y, rowvar=False)
        cov_inv = np.linalg.pinv(cov)
        diff = x - mu
        value = np.sqrt(diff.T @ cov_inv @ diff)
        return value
    

    @staticmethod
    def variance_inflation_factor(df:pd.DataFrame, features:list):
        """
        Compute the Variance Inflation Factor (VIF) for selected features.

        VIF quantifies multicollinearity among predictors. Higher values
        indicate redundancy among features, which can distort model estimates.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe containing features to assess.
        features : list
            List of feature column names for VIF computation.

        Returns
        -------
        pd.DataFrame
            DataFrame with VIF values and Anderson (2022) interpretative scale:
            - '< 1.8': No Multicollinearity
            - '1.8 to 5': Moderate
            - '5 to 10': Potential Multicollinearity
            - '> 10': Strong Multicollinearity
        """
        dff = df[features]
        dff = pd.get_dummies(dff, drop_first=True)
        dff = dff.apply(pd.to_numeric, errors='coerce')
        dff = dff.dropna()
        dff = dff.astype(float)

        vif_df = pd.DataFrame()
        vif_df['Variable'] = dff.columns
        vif_df['VIF'] = [
            round(outliers_influence.variance_inflation_factor(dff.values, i),3)
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
        """
        Compute pairwise correlation coefficients between selected features.

        Supports automatic encoding and type conversion for categorical variables
        if `numeric` is set to True. This function calculates Pearson correlations.

        Parameters
        ----------
        df : pd.DataFrame
            Dataset containing the features to analyze.
        features : list
            List of feature names to include in the correlation computation.
        numeric : bool, optional
            If True, encodes categorical features numerically for correlation.
            Default is False.

        Returns
        -------
        pd.DataFrame
            DataFrame with all unique feature pairs and their correlation values.
        """
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


    @staticmethod
    def plot_correlation(df:pd.DataFrame=None, x:list|str=None, y:list|str=None, color:list|str=None,
                         x_str:str=None, y_str:str=None, width:int=700, height:int=600):
        if isinstance(x, str):
            x_str = x
            x = df[x]
        if isinstance(y, str):
            y_str = y
            y = df[y]
        if isinstance(color, str):
            color = df[color]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=x, y=y, mode='markers',
            marker=dict(color=color),
        ))

        fig = plotly_main_layout(
            fig, title=f'Corr({x_str}, {y_str})', x=x_str, y=y_str,
            width=width, height=height
        )

        return fig

if __name__ == '__main__':
    ...