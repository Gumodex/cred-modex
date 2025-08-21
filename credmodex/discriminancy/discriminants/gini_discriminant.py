import sys
import os
import random
from typing import Literal
import warnings

import pandas as pd

import plotly.graph_objects as go

sys.path.append(os.path.abspath('.'))
from credmodex.utils.design import *
from credmodex.rating import *


__all__ = [
    'GINI_Discriminant',
]


class GINI_Discriminant():
    def __init__(self, df:pd.DataFrame=None, target:str=None, features:str|list[str]=None,
                 suppress_warnings:bool=False):
        self.df = df
        self.target = target
        self.suppress_warnings = suppress_warnings
        
        if isinstance(features,str):
            features = [features]
        self.features = features



    def value(self, col:str=None, is_continuous:bool=False, max_n_bins:int=30, force_discrete:bool=False, percent:bool=True, final_value:bool=False):
        if col is None:
            if ('score' in self.features):
                col = 'score'
            else:
                try: col = random.choice(self.features)
                except: raise ValueError("A column (col) must be provided")

        if pd.api.types.is_datetime64_any_dtype(self.df[col]):
            return None

        if (is_continuous) or (self.df[col].dtype == 'float') and (not force_discrete):
            binning = CH_Binning(max_n_bins=max_n_bins)
            binning.fit(self.df[col].dropna(), y=self.df[self.df[col].notna()][self.target])

            binning = binning.transform(self.df[col], metric="bins")
            self.df['bins'] = binning
        else:
            # Use categorical value counts
            self.df['bins'] = self.df[col]
    
        dff = self.df.groupby(['bins', self.target], observed=False).size().unstack(fill_value=0)
        dff.columns = ['Good', 'Bad']
        dff = dff[dff.index != 'Missing']
        dff['Total'] = dff['Good'] + dff['Bad']

        dff['Odds'] = (dff['Good'] / dff['Bad']).round(2)
        dff['Rate'] = (dff['Bad'] / dff['Total']).round(4)
        dff = dff.sort_values(by='Rate', ascending=False)

        total = dff['Total'].sum()
        total_B = dff['Bad'].sum()
        dff['Perfect'] = (dff['Total'].cumsum() / total_B).apply(lambda x: x if (x <= 1) else 1)
        
        dff['Good Cumul.'] = dff['Good'].cumsum() / dff['Good'].sum()
        dff['Bad Cumul.'] = dff['Bad'].cumsum() / dff['Bad'].sum()
        dff['Total Cumul.'] = dff['Total'].cumsum() / dff['Total'].sum()

        dff = dff.reset_index()

        for i in range(len(dff)):
            fg_i = dff.loc[i, 'Good Cumul.']
            fb_i = dff.loc[i, 'Bad Cumul.']
            
            if i == 0:
                product = fg_i * fb_i
            else:
                fg_prev = dff.loc[i - 1, 'Good Cumul.']
                fb_prev = dff.loc[i - 1, 'Bad Cumul.']
                product = (fg_i + fg_prev) * (fb_i - fb_prev)
            
            dff.loc[i, 'Product'] = product

        dff['Lift'] = dff['Bad Cumul.'] / dff['Total Cumul.']

        dff = dff.set_index('bins')
        gini_coeff = (1 - dff['Product'].sum()).round(3)

        if percent:
            for column in ['Rate', 'Perfect', 'Good Cumul.', 'Bad Cumul.', 'Total Cumul.', 'Product']:
                dff[column] = (100* dff[column]).round(4)
        else:
            dff = dff.round(4)

        if final_value:
            if percent: return round(100*gini_coeff,2)
            else: return round(gini_coeff,4)

        return dff


    def table(self, percent:bool=True):
        columns = self.df.columns.to_list()
        columns = [
            col for col in self.df.columns
            if col != self.target and not pd.api.types.is_datetime64_any_dtype(self.df[col])
            and (col in self.features)
        ]

        gini_df = pd.DataFrame(
            index=columns,
            columns=['Gini']
        )
        for col in columns:
            try:
                df = self.value(col=col, final_value=True, percent=percent)
                gini_df.loc[col,'Gini'] = df
            except:
                if not getattr(self, 'suppress_warnings', False):
                    warnings.warn(
                        '<log: column {col} discharted ({e})>',
                        category=UserWarning
                    )

        return gini_df
    

    def plot(self, col:str=None, method:Literal['gini','cap','lift']='lorenz gini', 
             max_n_bins:int=30, force_discrete:bool=False, width:int=700, height:int=600):
        if col is None:
            if ('score' in self.features):
                col = 'score'
            else:
                try: col = random.choice(self.features)
                except: raise ValueError("A column (col) must be provided")

        method = method.strip().lower()
        dff = self.value(col=col, max_n_bins=max_n_bins, force_discrete=force_discrete, percent=True)
        D = (100 - dff['Product'].sum()).round(3)

        if ('lor' in method) or ('gini' in method) or ('roc' in method) or ('auc' in method):
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[0]+dff['Good Cumul.'].to_list(), y=[0]+dff['Bad Cumul.'].to_list(),
                marker=dict(color='black'), name=col
            ))
            fig.add_trace(go.Scatter(
                x=[0,0,100], y=[0,100,100], name='Perfect',
                mode='lines', line=dict(dash='dash', color='rgb(26, 26, 26)')
            ))
            fig.add_trace(go.Scatter(
                x=[0,100], y=[0,100], name='Random',
                mode='lines', line=dict(dash='dash', color='rgb(218, 62, 86)')
            ))
            plotly_main_layout(fig, title=f'Lorenz & Gini | D = {D}',
                x='Cumulative Goods', y='Cumulative Bads', x_range=[-0.5,101], y_range=[-0.5,101],
                width=width, height=height
            )

        if ('cap' in method) or ('accur' in method) or ('ratio' in method):
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[0]+dff['Total Cumul.'].to_list(), y=[0]+dff['Bad Cumul.'].to_list(),
                marker=dict(color='black'), name=col
            ))
            fig.add_trace(go.Scatter(
                x=[0]+dff['Total Cumul.'].to_list(), y=[0]+dff['Perfect'].to_list(), name='Perfect',
                mode='lines', line=dict(dash='dash', color='rgb(26, 26, 26)')
            ))
            fig.add_trace(go.Scatter(
                x=[0,100], y=[0,100], name='Random',
                mode='lines', line=dict(dash='dash', color='rgb(218, 62, 86)')
            ))
            plotly_main_layout(fig, title=f'Lorenz & Gini | D = {D}',
                x='Cumulative Total', y='Cumulative Bads', x_range=[-0.5,101], y_range=[-0.5,101],
                width=width, height=height
            )

        if ('lift' in method):
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dff['Total Cumul.'].to_list(), y=dff['Lift'].to_list(),
                marker=dict(color='black'), name=col
            ))
            fig.add_trace(go.Scatter(
                x=[0,100], y=[1,1], name='Random',
                mode='lines', line=dict(dash='dash', color='rgb(218, 62, 86)')
            ))
            plotly_main_layout(fig, title=f'Lift Chart | D = {D}',
                x='Cumulative Total', y='Lift', x_range=[dff['Total Cumul.'].min()-0.1,101], y_range=[0.5,dff['Lift'].max()+0.1],
                width=width, height=height
            )

        try: return fig
        except: return None