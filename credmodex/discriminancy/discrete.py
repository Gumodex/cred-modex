import sys
import os
import warnings
import itertools

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




class IV_Discriminant():
    def __init__(self, df:pd.DataFrame=None, target:str=None, features:list[str]=None):
        self.df = df
        self.target = target
        self.features = features
        assert set(self.df[self.target].unique()) == {0, 1}, "Target must be binary 0/1"


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





class PSI_Discriminant():
    def __init__(self, df:pd.DataFrame=None, target:str=None, features:list[str]=None):
        self.df = df
        self.target = target
        self.features = features


    def value(self, col:str=None, percent_shift:float=0.8, is_continuous:bool=False, final_value:bool=False):
        # Split data using iloc
        split_index = int(len(self.df) * percent_shift)
        self.train = self.df.iloc[:split_index]
        self.test = self.df.iloc[split_index:]

        if pd.api.types.is_datetime64_any_dtype(self.df[col]):
            return None

        if (is_continuous) or (self.df[col].dtype == 'float'):
            # Create bins based on training data
            binning = OptimalBinning(name=col, dtype="numerical", max_n_bins=10)
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
    

    def table(self, percent_shift:float=0.8):
        columns = self.df.columns.to_list()
        columns = [col for col in columns if col != self.target]

        psi_df = pd.DataFrame(
            index=columns,
            columns=['PSI','ANDERSON (2022)']
        )
        for col in columns:
            try:
                df = self.value(col=col, percent_shift=percent_shift)
                psi_df.loc[col,'PSI'] = df.loc['Total','PSI'].round(4)
                psi_df.loc[col,'ANDERSON (2022)'] = df.loc['Total','ANDERSON (2022)']
            except:
                print(f'<log: column {col} discharted>')

        return psi_df
    

    def plot(self, col:str=None, percent_shift:float=0.8, discrete:bool=False, width:int=900, height:int=450):
        dff = self.value(col=col, percent_shift=percent_shift)
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

            plotly_main_layout(fig, title='Population Stability Analysis', x=col, y='freq', width=width, height=height)

            return fig

        except:
            return





class Correlation():
    def __init__(self, df:pd.DataFrame=None, target:str=None, features:list[str]=None):
        self.df = df
        self.target = target
        self.features = features

    
    def VIF(self):
        dff = self.df[self.features]
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
    

    def correlation(self, numeric:bool=False):
        dff = self.df[self.features]
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




if __name__ == "__main__":
    print(
        Correlation(df, target='over', features=['idade','score_scr','UF']).correlation()
    )
    Correlation(df, target='over', features=['idade','score_scr'])
    