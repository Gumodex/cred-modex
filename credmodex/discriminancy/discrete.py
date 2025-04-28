import sys
import os
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

sys.path.append(os.path.abspath('.'))
from credmodex.utils.design import *

df = pd.read_csv(r'C:\Users\gustavo.filho\Documents\Python\Modules\Credit Risk\test\df.csv')




class IV_Discriminant():
    def __init__(self, df:pd.DataFrame=None, target:str=None, features:list[str]=None):
        self.df = df
        self.target = target
        self.features = features


    def value(self, col:str=None, final_value:bool=False):
        if col is None:
            raise ValueError("A column (col) must be provided")
        
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
        columns = [col for col in columns if col != self.target]

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
    




class KS_Discriminant():
    def __init__(self, df:pd.DataFrame=None, target:str=None, features:list[str]=None):
        self.df = df
        self.target = target
        self.features = features


    def value(self, col:str=None, final_value:bool=False, sort:str=None, bad_:int=1, plot_:bool=False):
        if col is None:
            raise ValueError("A column (col) must be provided")
        
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
        columns = [col for col in columns if col != self.target]
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
    

    def plot(self, col:str=None, sort:str=None, graph_library:str='plotly', width:int=900, height:int=450):
        if col is None:
            raise ValueError("A column (col) must be provided")
        
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







if __name__ == "__main__":
    print(
        KS_Discriminant(df, target='over', features=['idade','score_scr']).plot(col='idade', sort='ascending', graph_library='plotly')
    )
    KS_Discriminant(df, target='over', features=['idade','score_scr']).plot(col='idade', sort='ascending', graph_library='plotly').show()