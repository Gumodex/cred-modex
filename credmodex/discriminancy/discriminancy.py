import sys
import os
import warnings
import itertools
from typing import Literal, Optional

import pandas as pd
import numpy as np

from optbinning import OptimalBinning

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.figure_factory as ff

import scipy.stats
import statsmodels.stats
import statsmodels.stats.outliers_influence

sys.path.append(os.path.abspath('.'))
from credmodex.utils.design import *

df = pd.read_csv(r'C:\Users\gustavo.filho\Documents\Python\Modules\Credit Risk\test\df.csv')


__all__ = [
    'IV_Discriminant', 
    'KS_Discriminant',
    'PSI_Discriminant',
    'GINI_LORENZ_Discriminant',
    'CHI2_Discriminant',
]


class IV_Discriminant():
    def __init__(self, df:pd.DataFrame=None, target:str=None, features:list[str]=None):
        self.df = df
        self.target = target
        self.features = features
        assert set(self.df[self.target].unique()) == {0, 1}, "Target must be binary 0/1"


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





class PSI_Discriminant():
    def __init__(self, df:pd.DataFrame=None, target:str=None, features:list[str]=None):
        self.df = df
        self.target = target
        self.features = features


    def value(self, col:str=None, percent_shift:float=0.8, is_continuous:bool=False, max_n_bins:int=10, final_value:bool=False):
        if col is None:
            if ('score' in self.features):
                col = 'score'
            else:
                try: col = random.choice(self.features)
                except: raise ValueError("A column (col) must be provided")

        split_index = int(len(self.df) * percent_shift)
        self.train = self.df.iloc[:split_index]
        self.test = self.df.iloc[split_index:]

        if pd.api.types.is_datetime64_any_dtype(self.df[col]):
            return None

        if (is_continuous) or (self.df[col].dtype == 'float'):
            # Create bins based on training data
            binning = OptimalBinning(name=col, dtype="numerical", max_n_bins=max_n_bins)
            binning.fit(self.train[col].dropna(), y=self.train[self.train[col].notna()][self.target])

            # Apply binning to train and test sets
            train_binned = binning.transform(self.train[col], metric="bins")
            test_binned = binning.transform(self.test[col], metric="bins")

            # Convert to categorical for grouping
            train = pd.Series(train_binned).value_counts(normalize=True).sort_index().rename("Reference")
            test = pd.Series(test_binned).value_counts(normalize=True).sort_index().rename("Posterior")
        else:
            # Use categorical value counts
            train = self.train[col].value_counts(normalize=True).rename('Reference')
            test = self.test[col].value_counts(normalize=True).rename('Posterior')

        # Combine and handle zero issues
        dff = pd.concat([train, test], axis=1).fillna(0.0001).round(4)
        dff = dff[dff.index != 'Missing']

        # Calculate PSI
        dff['PSI'] = round((dff['Reference'] - dff['Posterior']) * np.log(dff['Reference'] / dff['Posterior']), 4)
        dff['PSI'] = dff['PSI'].apply(lambda x: 0 if x in {np.nan, np.inf} else x)

        # Total PSI
        dff.loc['Total'] = dff.sum(numeric_only=True).round(4)

        # Anderson-style classification
        anderson_conditions = [
            (dff['PSI'] <= 0.10),
            (dff['PSI'] > 0.10) & (dff['PSI'] <= 0.25),
            (dff['PSI'] > 0.25) & (dff['PSI'] <= 1.00),
            (dff['PSI'] > 1.00),
        ]
        anderson_values = ['Green', 'Yellow', 'Red', 'Accident']
        dff['ANDERSON (2022)'] = np.select(anderson_conditions, anderson_values, '-')

        if final_value:
            return dff.loc['Total', 'PSI']

        return dff
    

    def table(self, percent_shift:float=0.8, max_n_bins:int=10):
        columns = self.df.columns.to_list()
        columns = [col for col in columns if col != self.target]

        psi_df = pd.DataFrame(
            index=columns,
            columns=['PSI','ANDERSON (2022)']
        )
        for col in columns:
            try:
                df = self.value(col=col, percent_shift=percent_shift, max_n_bins=max_n_bins)
                psi_df.loc[col,'PSI'] = df.loc['Total','PSI'].round(4)
                psi_df.loc[col,'ANDERSON (2022)'] = df.loc['Total','ANDERSON (2022)']
            except:
                print(f'<log: column {col} discharted>')

        return psi_df
    

    def plot(self, col:str=None, percent_shift:float=0.8, discrete:bool=False, max_n_bins:int=10, width:int=900, height:int=450):
        if col is None:
            if ('score' in self.features):
                col = 'score'
            else:
                try: col = random.choice(self.features)
                except: raise ValueError("A column (col) must be provided")

        dff = self.value(col=col, percent_shift=percent_shift, max_n_bins=max_n_bins)
        psi = dff.loc['Total','ANDERSON (2022)']
        if dff is None: 
            return
        
        if (discrete) or (self.df[col].dtype in {'float', 'int'}):
            dff = dff[dff.index != 'Total']

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x = dff.index, y = dff['Reference'],
                name = f'Train | {100* (percent_shift):.1f}%',
                marker=dict(color='rgb(218, 139, 192)')
            ))
            fig.add_trace(go.Bar(
                x = dff.index, y = dff['Posterior'],
                name = f'Test | {100* (1-percent_shift):.1f}%',
                marker=dict(color='rgb(170, 98, 234)')
            ))

            plotly_main_layout(fig, title='Population Stability Analysis', x=col, y='freq', width=width, height=height)

            return fig

        try:
            fig = go.Figure()
            train_plot = ff.create_distplot(
                    hist_data=[self.train[col].dropna()],
                    group_labels=['distplot'],
                )['data'][1]
            train_plot['marker']['color'] = 'rgb(218, 139, 192)'
            train_plot['fillcolor'] = 'rgba(218, 139, 192, 0.2)'
            train_plot['fill'] = 'tozeroy'
            train_plot['name'] = f'Train | {100* (percent_shift):.1f}%'
            train_plot['showlegend'] = True

            test_plot = ff.create_distplot(
                    hist_data=[self.test[col].dropna()],
                    group_labels=['distplot']
                )['data'][1]
            test_plot['marker']['color'] = 'rgb(170, 98, 234)'
            test_plot['fillcolor'] = 'rgba(170, 98, 234, 0.2)'
            test_plot['fill'] = 'tozeroy'
            test_plot['name'] = f'Test | {100* (1-percent_shift):.1f}%'
            test_plot['showlegend'] = True

            fig.add_trace(test_plot)
            fig.add_trace(train_plot)

            plotly_main_layout(fig, title=f'Population Stability Analysis | PSI = {psi}', x=col, y='freq', width=width, height=height)

            return fig

        except:
            return





class GINI_LORENZ_Discriminant():
    def __init__(self, df:pd.DataFrame=None, target:str=None, features:list[str]=None):
        self.df = df
        self.target = target
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
            binning = OptimalBinning(name=col, dtype="numerical", max_n_bins=max_n_bins)
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
                print(f'<log: column {col} discharted>')

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





class CHI2_Discriminant():
    def __init__(self, df:pd.DataFrame=None, target:str=None, features:list[str]=None):
        self.df = df
        self.target = target
        self.features = features


    def value(self, col:str=None, percent_shift:float=None, final_value:bool=False):
        if col is None:
            if ('score' in self.features):
                col = 'score'
            else:
                try: col = random.choice(self.features)
                except: raise ValueError("A column (col) must be provided")

        self.observed = df[df.index <= (len(df)*percent_shift)]
        self.expected = df[df.index > (len(df)*percent_shift)]

        observed = self.observed.groupby([col, self.target], observed=False).size().unstack(fill_value=0)
        observed.columns = ['O(Good)', 'O(Bad)']
        observed['O(Total)'] = observed['O(Good)'] + observed['O(Bad)']
        observed['O(Odd)'] = round(observed['O(Good)'] / observed['O(Bad)'],2)

        expected = self.expected.groupby([col, self.target], observed=False).size().unstack(fill_value=0)
        expected.columns = ['E(Good)', 'E(Bad)']

        dff = pd.concat([observed, expected], axis=1)
        dff['Chi-Square'] = (dff['O'] - dff['E'])**2 / dff['E']
        

        return dff






if __name__ == "__main__":
    ...

    