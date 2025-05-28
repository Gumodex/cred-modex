import sys
import os
import random
from typing import Literal

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import plotly.graph_objects as go

sys.path.append(os.path.abspath('.'))
from credmodex.utils.design import *
from credmodex.rating import *


__all__ = [
    'KS_Discriminant',
]


class KS_Discriminant():
    def __init__(self, df:pd.DataFrame=None, target:str=None, features:str|list[str]=None):
        self.df = df
        self.target = target

        if isinstance(features,str):
            features = [features]
        self.features = features


    def value(self, col:str=None, final_value:bool=False, sort:str=None, bad_:int=1, plot_:bool=False):
        if col is None:
            if ('score' in self.features):
                col = 'score'
            else:
                try: col = random.choice(self.features)
                except: raise ValueError("A column (col) must be provided")

        if pd.api.types.is_datetime64_any_dtype(self.df[col]):
            return None

        df = self.df.copy(deep=True)[[col, self.target]]
        volumetry = df[df[col].notna()].groupby(by=col, observed=False)[self.target].count().astype(float).sum()
        df_ks = pd.DataFrame(df.groupby(by=col, observed=False)[self.target].count().astype(float))

        if (sort == 'ascending') or (df[col].dtype == 'float64'):
            df_ks = df_ks.sort_values(by=col, ascending=True)

        if (bad_ == 1):
            df_ks['Bad'] = df.groupby(by=col, observed=False)[self.target].sum()
        elif (bad_ == 0): 
            df_ks['Bad'] = df_ks[self.target] - df.groupby(by=col)[self.target].sum()
        total_bad = df_ks['Bad'].sum()
        total_good = df_ks[self.target].sum() - df_ks['Bad'].sum()

        df_ks['% Bad'] = round(100* df_ks['Bad'] / df_ks[self.target],3)
        if (sort != 'ascending') and (df[col].dtype != 'float64'):
            df_ks = df_ks.sort_values(by='% Bad', ascending=False)

        df_ks['F (bad)'] = round(100* df_ks['Bad'].cumsum() / total_bad,3)
        df_ks['F (good)'] = round(100* (df_ks[self.target] - df_ks['Bad']).cumsum() / total_good,3)

        df_ks['KS'] = np.abs(df_ks['F (bad)'] - df_ks['F (good)'])
        try:
            KS = round(max(df_ks['KS']),4)
        except: return None

        del df_ks['% Bad']; del df_ks[self.target]; del df_ks['Bad']

        if final_value:
            try: return round(KS,3)
            except: return None

        if plot_:
            return df_ks, volumetry, KS
        
        return df_ks
    

    def table(self):
        columns = self.df.columns.to_list()
        columns = [col for col in columns if (col != self.target) and (col in self.features)]
        KS_Value = pd.DataFrame(
            index=columns,
            columns=['KS']
        )
        for col in columns:
            try:
                df_ks = self.value(col=col, final_value=False)
                ks_col = round(max(df_ks['KS']),4)
                KS_Value.loc[col,'KS'] = ks_col
            except:
                ...

        credit_scoring = [
            (KS_Value['KS'] < 20),
            (KS_Value['KS'] >= 20) & (KS_Value['KS'] <= 30),
            (KS_Value['KS'] >= 30) & (KS_Value['KS'] <= 40),
            (KS_Value['KS'] >= 40) & (KS_Value['KS'] <= 50),
            (KS_Value['KS'] >= 50) & (KS_Value['KS'] <= 60),
            (KS_Value['KS'] > 60),
        ]
        credit_scoring_values = ['Low', 'Acceptable', 'Good', 'Very Good', 'Excelent', 'Unusual']
        KS_Value['Credit Score'] = np.select(credit_scoring, credit_scoring_values, '-')

        behavioral = [
            (KS_Value['KS'] < 20),
            (KS_Value['KS'] >= 20) & (KS_Value['KS'] <= 30),
            (KS_Value['KS'] >= 30) & (KS_Value['KS'] <= 40),
            (KS_Value['KS'] >= 40) & (KS_Value['KS'] <= 50),
            (KS_Value['KS'] >= 50) & (KS_Value['KS'] <= 60),
            (KS_Value['KS'] > 60),
        ]
        behavioral_values = ['Low', 'Low', 'Low', 'Acceptable', 'Good', 'Excelent']
        KS_Value['Behavioral Score'] = np.select(behavioral, behavioral_values, '-')

        return KS_Value.sort_values(by='KS', ascending=False)
    

    def plot(self, col:str=None, sort:Literal[None,'ascendng']=None, 
             graph_library:Literal['plotly','matplotlib']='plotly', width:int=900, height:int=450):
        if col is None:
            if ('score' in self.features):
                col = 'score'
            else:
                try: col = random.choice(self.features)
                except: raise ValueError("A column (col) must be provided")
        
        df_ks, volumetry, KS = self.value(col=col, sort=sort, plot_=True)
        
        if graph_library == 'plotly':
            fig = go.Figure()
            fig.add_trace(trace=go.Scatter(
                x=df_ks.index, y=df_ks['F (bad)'], name=r'F (bad)',
                mode='lines+markers', line=dict(color='#e04c1a'), 
                marker=dict(size=6, color='#ffffff', line=dict(color='#e04c1a', width=2))
            ))
            fig.add_trace(trace=go.Scatter(
                x=df_ks.index, y=df_ks['F (good)'], name=r'F (good)',
                mode='lines+markers', line=dict(color='#3bc957'), 
                marker=dict(size=6, color='#ffffff', line=dict(color='#3bc957', width=2))
            ))
            x_ks = df_ks[df_ks['KS'] == max(df_ks['KS'])].index.values[0]
            y1_ks = df_ks[df_ks['KS'] == max(df_ks['KS'])]['F (good)'].values[0]
            y2_ks = df_ks[df_ks['KS'] == max(df_ks['KS'])]['F (bad)'].values[0]
            fig.add_trace(trace=go.Scatter(
                x=[x_ks, x_ks], y=[y1_ks, y2_ks], name=f'KS = {KS:.2f}%',
                mode='lines+markers', line=dict(color='#080808'), 
                marker=dict(size=6, color='#ffffff', line=dict(color='#080808', width=2))
            ))
            return plotly_main_layout(
                fig, title=f'KS | {col} (Metric: {self.target} | V: {volumetry:.0f})', 
                x=col, y='Cumulative Percentage', height=height, width=width,
                )

        elif graph_library == 'matplotlib':
            fig, ax = plt.subplots()

            fig, ax = matplotlib_main_layout(
                fig, ax, title=f'KS | {col} (Metric: {self.target} | V: {volumetry:.0f})', 
                x=col, y='Cumulative Percentage', height=height, width=width,
            )
            
            ax.plot(df_ks.index, df_ks['F (bad)'], label=r'F (bad)', color='#e04c1a', marker='o', markersize=6, linewidth=2, markerfacecolor='white')
            ax.plot(df_ks.index, df_ks['F (good)'], label=r'F (good)', color='#3bc957', marker='o', markersize=6, linewidth=2, markerfacecolor='white')

            x_ks = df_ks[df_ks['KS'] == max(df_ks['KS'])].index.values[0]
            y1_ks = df_ks[df_ks['KS'] == max(df_ks['KS'])]['F (good)'].values[0]
            y2_ks = df_ks[df_ks['KS'] == max(df_ks['KS'])]['F (bad)'].values[0]
            ax.plot([x_ks, x_ks], [y1_ks, y2_ks], label=f'KS = {max(df_ks["KS"]):.2f}%', color='#080808', marker='o', markersize=6, linewidth=2, markerfacecolor='white')

            ax.legend()
            return fig, ax

