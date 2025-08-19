import warnings
import random

import pandas as pd
import numpy as np


__all__ = [
    'IV_Discriminant', 
]


class IV_Discriminant():
    def __init__(self, df:pd.DataFrame=None, target:str=None, features:str|list[str]=None):
        self.df = df.copy(deep=True)
        self.target = target

        self.df = self.df[self.df[self.target].notna()]

        assert set(self.df[self.target].unique()) == {0, 1}, "Target must be binary 0/1"
        
        if isinstance(features,str):
            features = [features]
        self.features = features


    def value(self, col:str=None, final_value:bool=False):
        if col is None:
            if ('score' in self.features):
                col = 'score'
            else:
                try: col = random.choice(self.features)
                except: raise ValueError("A column (col) must be provided")
        
        if pd.api.types.is_datetime64_any_dtype(self.df[col]):
            return None

        woe_iv_df = self.df.groupby([col, self.target], observed=False).size().unstack(fill_value=0)
        woe_iv_df.columns = ['Good', 'Bad']
        woe_iv_df.loc['Total'] = woe_iv_df.sum()

        woe_iv_df['Total'] = woe_iv_df['Good'] + woe_iv_df['Bad']

        woe_iv_df['Good (col)'] = woe_iv_df['Good'] / woe_iv_df.loc['Total', 'Good']
        woe_iv_df['Bad (col)'] = woe_iv_df['Bad'] / woe_iv_df.loc['Total', 'Bad']

        woe_iv_df['Good (row)'] = woe_iv_df['Good']/woe_iv_df['Total']
        woe_iv_df['Bad (row)'] = woe_iv_df['Bad']/woe_iv_df['Total']

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered in log")

            woe_iv_df['WOE'] = np.log(woe_iv_df['Good (col)'] / woe_iv_df['Bad (col)'])
            woe_iv_df['IV'] = (woe_iv_df['Good (col)'] - woe_iv_df['Bad (col)']) * woe_iv_df['WOE']
            woe_iv_df['IV'] = woe_iv_df['IV'].apply(lambda x: round(x,6))
            woe_iv_df['B/M'] = round(woe_iv_df['Good (row)']/woe_iv_df['Bad (row)'],2)

            woe_iv_df = woe_iv_df[(woe_iv_df['IV'] != np.inf) & (woe_iv_df['IV'] != -np.inf)]

            woe_iv_df.loc['Total','IV'] = woe_iv_df.loc[:,'IV'].sum()
            woe_iv_df.loc['Total','WOE'] = np.nan
            woe_iv_df.loc['Total','B/M'] = np.nan

        woe_iv_df['Good (col)'] = woe_iv_df['Good (col)'].apply(lambda x: round(100*x,2))
        woe_iv_df['Bad (col)'] = woe_iv_df['Bad (col)'].apply(lambda x: round(100*x,2))
        woe_iv_df['Good (row)'] = woe_iv_df['Good (row)'].apply(lambda x: round(100*x,2))
        woe_iv_df['Bad (row)'] = woe_iv_df['Bad (row)'].apply(lambda x: round(100*x,2))

        if final_value:
            try: return round(woe_iv_df.loc['Total','IV'],3)
            except: return None

        return woe_iv_df
    

    def table(self):
        columns = self.df.columns.to_list()
        columns = [col for col in columns if (col != self.target) and (col in self.features)]

        iv_df = pd.DataFrame(
            index=columns,
            columns=['IV']
        )
        for col in columns:
            try:
                df = self.value(col=col)
                iv_df.loc[col,'IV'] = round(df.loc['Total','IV'],6)
            except:
                print(f'<log: column {col} discharted>')

        siddiqi_conditions = [
            (iv_df['IV'] < 0.03),
            (iv_df['IV'] >= 0.03) & (iv_df['IV'] <= 0.1),
            (iv_df['IV'] >= 0.1) & (iv_df['IV'] <= 0.3),
            (iv_df['IV'] > 0.3) & (iv_df['IV'] <= 0.5),
            (iv_df['IV'] > 0.5),
        ]
        siddiqi_values = ['No Discr.' ,'Weak', 'Moderate', 'Strong', 'Super Strong']
        iv_df['SIDDIQI (2006)'] = np.select(siddiqi_conditions, siddiqi_values, '-')

        thomas_conditions = [
            (iv_df['IV'] < 0.03),
            (iv_df['IV'] >= 0.03) & (iv_df['IV'] <= 0.1),
            (iv_df['IV'] >= 0.1) & (iv_df['IV'] <= 0.25),
            (iv_df['IV'] > 0.25),
        ]
        thomas_values = ['No Discr.', 'Weak', 'Moderate', 'Strong']
        iv_df['THOMAS (2002)'] = np.select(thomas_conditions, thomas_values, '-')

        anderson_conditions = [
            (iv_df['IV'] < 0.05),
            (iv_df['IV'] >= 0.05) & (iv_df['IV'] <= 0.1),
            (iv_df['IV'] >= 0.1) & (iv_df['IV'] <= 0.3),
            (iv_df['IV'] >= 0.3) & (iv_df['IV'] <= 0.5),
            (iv_df['IV'] >= 0.5) & (iv_df['IV'] <= 1.0),
            (iv_df['IV'] > 1.0),
        ]
        anderson_values = ['No Discr.', 'Weak', 'Moderate', 'Strong', 'Super Strong', 'Overpredictive']
        iv_df['ANDERSON (2022)'] = np.select(anderson_conditions, anderson_values, '-')

        return iv_df.sort_values(by='IV', ascending=False)