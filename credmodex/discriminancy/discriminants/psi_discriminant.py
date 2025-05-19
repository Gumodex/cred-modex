import sys
import os
import random

import pandas as pd
import numpy as np

import plotly.graph_objects as go
import plotly.figure_factory as ff

sys.path.append(os.path.abspath('.'))
from credmodex.utils.design import *
from credmodex.rating import *


__all__ = [
    'PSI_Discriminant',
]


class PSI_Discriminant():
    def __init__(self, df:pd.DataFrame=None, target:str=None, features:list[str]=None):
        self.df = df
        self.target = target
        if isinstance(features,str):
            features = [features]
        self.features = features


    def value(self, col:str=None, percent_shift:float=0.8, is_continuous:bool=False, max_n_bins:int=10, 
              final_value:bool=False, add_min_max:list=[None, None]):
        if col is None:
            if ('score' in self.features):
                col = 'score'
            else:
                try: col = random.choice(self.features)
                except: raise ValueError("A column (col) must be provided")

        split_index = int(len(self.df) * percent_shift)
        self.train = self.df.iloc[:split_index]
        self.test = self.df.iloc[split_index:]

        if (add_min_max[0] is not None) and (add_min_max[-1] is not None):
            self.train = pd.concat([
                self.train, 
                pd.DataFrame([add_min_max[0]], columns=[col]), 
                pd.DataFrame([add_min_max[-1]], columns=[col])
            ])
            self.test = pd.concat([
                self.test, 
                pd.DataFrame([add_min_max[0]], columns=[col]), 
                pd.DataFrame([add_min_max[-1]], columns=[col])
            ])




        if pd.api.types.is_datetime64_any_dtype(self.df[col]):
            return None

        if (is_continuous) or (self.df[col].dtype == 'float'):
            # Create bins based on training data
            binning = CH_Binning(max_n_bins=max_n_bins)
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
        columns = [col for col in columns if (col != self.target) and (col in self.features)]

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
    

    def plot(self, col:str=None, percent_shift:float=0.8, discrete:bool=False, max_n_bins:int=10, width:int=900, height:int=450,
             add_min_max:list=[None, None]):
        if col is None:
            if ('score' in self.features):
                col = 'score'
            else:
                try: col = random.choice(self.features)
                except: raise ValueError("A column (col) must be provided")

        dff = self.value(col=col, percent_shift=percent_shift, max_n_bins=max_n_bins,
                         add_min_max=add_min_max)
        psi = dff.loc['Total','ANDERSON (2022)']
        if dff is None: 
            return

        try:
            if (discrete == True):
                raise TypeError
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
            try:
                if (discrete) or (~pd.api.types.is_numeric_dtype(self.df[col])):
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

                    plotly_main_layout(fig, title=f'Population Stability Analysis | PSI = {psi}', x=col, y='freq', width=width, height=height)

                    return fig
            except:
                return

