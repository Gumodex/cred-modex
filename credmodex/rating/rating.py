import sys
import os
import warnings
import inspect
from typing import Union, Literal

import pandas as pd
import numpy as np

import plotly
import plotly.express as px
import plotly.graph_objects as go

sys.path.append(os.path.abspath('.'))
from credmodex.rating.binning import CH_Binning
from credmodex.discriminancy.discriminants.ks_discriminant import KS_Discriminant
from credmodex.utils import *


__all__ = [
    'Rating'
]


class Rating():
    def __init__(self, df:pd.DataFrame=None, features:Union[list[str],str]='score', target:str=None, time_col:str=None, model:type=CH_Binning(max_n_bins=15), 
                 type:str='score', optb_type:str='transform', doc:str=None, suppress_warnings:bool=False, name:str=None):
        
        if isinstance(features,str):
            features = [features]
        if (features is None):
            features = df.columns.to_list()
        features = [f for f in features 
                    if f in df.columns 
                    and f != target 
                    and f != time_col
                    and f != 'id']
        
        if (df is None):
            raise ValueError("DataFrame cannot be None. Input a DataFrame.")
        if (model is None):
            model = lambda df: df  # Default model that does nothing
        if model == 'CH_Binning':
            model = CH_Binning(max_n_bins=15)

        self.model = model
        self.df = df.copy(deep=True)
        self.doc = doc
        self.optb_type = optb_type
        self.name = name

        self.id = 'id'
        self.time_col = time_col
        self.features = features
        self.target = target
        self.type = type
        self.suppress_warnings = suppress_warnings
        
        self.train_test_()

        if callable(self.model):
            self.model_code = inspect.getsource(self.model)
        else:
            self.model_code = None

        if callable(self.doc):
            self.doc = inspect.getsource(self.doc)
        else:
            self.doc = None

        if (self.type == 'score'):
            try:
                self.fit_predict_score()
            except Exception as e:
                print("ERROR IN fit_predict_score:", str(e))
                raise

    
    def train_test_(self):
        try:
            self.train = self.df[self.df['split'] == 'train']
            self.test = self.df[self.df['split'] == 'test']

            transformed_features = [col for col in self.df.columns if col not in ['split', 'target', self.time_col, 'id']]
            self.features = transformed_features

            self.X_train = self.df[self.df['split'] == 'train'][self.features]
            self.X_test = self.df[self.df['split'] == 'test'][self.features]
            self.y_train = self.df[self.df['split'] == 'train'][self.target]
            self.y_test = self.df[self.df['split'] == 'test'][self.target]
        except:
            if not getattr(self, 'suppress_warnings', False):
                warnings.warn(
                    'No column ["split"] was found, therefore, the whole `df` will be used in training and testing',
                    category=UserWarning
                )
            self.train = self.df
            self.test = self.df

            transformed_features = [col for col in self.df.columns if col not in ['split', 'target', self.time_col, 'id']]
            self.features = transformed_features

            self.X_train = self.df[self.features]
            self.X_test = self.df[self.features]
            self.y_train = self.df[self.target]
            self.y_test = self.df[self.target]


    def fit_predict_score(self):
        
        if ('score' not in self.df.columns):
            if not getattr(self, 'suppress_warnings', False):
                warnings.warn(
                    '``score`` must be provided in df.columns to perform fit_predict_score\nCurrently, no fitting is being made!\nUse ``model = None`` only if ["rating"] column already exists',
                    category=UserWarning
                )

        if callable(self.model):
            self.df = self.model(self.df)
            return

        self.model.fit(self.train['score'], self.y_train)

        optb_type = self.optb_type.lower().strip() if isinstance(self.optb_type, str) else None

        if optb_type and ('trans' in optb_type):
            if not hasattr(self.model, 'transform'):
                raise AttributeError("Model has no `transform` method.")

            try: transformed = self.model.transform(self.df['score'], metric='bins')
            except TypeError: transformed = self.model.transform(self.df['score'])

            self.df['rating'] = transformed

            try:
                bin_table = self.model.binning_table.build()
                bins = list(bin_table['Bin'].unique())
            except Exception as e:
                bins = self.df.groupby('rating')[self.target].mean().sort_values(ascending=True)
                bins = list(bins.index)

            bin_map = Rating.map_to_alphabet_(bins)
            self.bins = bin_map
            self.df['rating'] = self.df['rating'].map(bin_map)
            return

        if optb_type is None:
            return
        raise ValueError(f"Unknown optb_type: {self.optb_type}")
    


    def predict(self, df_):
        
        if ('score' not in df_.columns):
            if not getattr(self, 'suppress_warnings', False):
                warnings.warn(
                    '["score"] must be provided in df.columns to predict something',
                    category=UserWarning
                )

        if callable(self.model):
            df_ = self.model(df_)
            return

        optb_type = self.optb_type.lower().strip() if isinstance(self.optb_type, str) else None

        if optb_type and ('trans' in optb_type):
            if not hasattr(self.model, 'transform'):
                raise AttributeError("Model has no `transform` method.")

            try: transformed = self.model.transform(df_['score'], metric='bins')
            except TypeError: transformed = self.model.transform(df_['score'])

            df_['rating'] = transformed

            try:
                bin_table = self.model.binning_table.build()
                bins = list(bin_table['Bin'].unique())
            except Exception as e:
                bins = df_.groupby('rating')[self.target].mean().sort_values(ascending=True)
                bins = list(bins.index)

            bin_map = Rating.map_to_alphabet_(bins)
            self.bins = bin_map
            df_['rating'] = df_['rating'].map(bin_map)
            return df_

        if optb_type is None:
            return df_
        raise ValueError(f"Unknown optb_type: {self.optb_type}")
    

        
    @staticmethod
    def map_to_alphabet_(bin_list):
        valid_bins = [b for b in bin_list if b not in ['Special', 'Missing', '']]
        try: valid_bins = sorted(valid_bins, key=lambda x: float(x.split(',')[1].replace(')', '').replace('inf', '1e10')), reverse=True)
        except: ...
        bin_map = {bin_label: chr(65 + i) for i, bin_label in enumerate(valid_bins)}
        return bin_map
                

    def plot_stability_in_time(self, df:pd.DataFrame=None, initial_date:str=None, upto_date:str=None, col:str='rating', 
                               agg_func:str='mean', percent:bool=True, split:list|Literal['train','test','oot']=['train','test','oot'], 
                               sample:float=None, width=800, height=600, color_seq:px.colors=px.colors.sequential.Turbo,
                               stackgroup:bool|str=None, **kwargs):
        if (df is not None):
            dff = df.copy(deep=True)
        else:
            dff = self.df.copy(deep=True)

        if isinstance(split, str):
            split = [split]
        try:
            dff = dff[dff['split'].isin(split)]
        except:
            if ('split' not in dff.columns) and (not getattr(self, 'suppress_warnings', False)):
                warnings.warn(
                    'No column ["split"] was found, therefore, the whole `df` will be used in this method',
                    category=UserWarning
                )
            dff = dff.copy(deep=True)
            
        if (sample is not None):
            sample = np.abs(sample)
            if (sample <= 1):
                dff = dff.sample(frac=sample)
            elif (sample > 1):
                if (sample > len(dff)):
                    raise ValueError(f"Sample size {sample} is larger than the dataset size {len(dff)}.")
                dff = dff.sample(n=sample)
    
        if initial_date is not None:
            initial_date = pd.to_datetime(initial_date)
            dff = dff[dff[self.time_col] >= initial_date]

        if upto_date is not None:
            upto_date = pd.to_datetime(upto_date)
            dff = dff[dff[self.time_col] <= upto_date]

        ratings = sorted(dff[col].unique())
        sample_points = [i / (len(ratings) - 1) for i in range(len(ratings))]
        colors = plotly.colors.sample_colorscale(color_seq, sample_points) 

        if percent:
            dff = round(100* dff[[self.time_col, col, self.target]].pivot_table(index=col, columns=pd.to_datetime(dff[self.time_col]).dt.strftime('%Y-%m'), values=self.target, aggfunc=agg_func),2 )
        else:
            dff = round(dff[[self.time_col, col, self.target]].pivot_table(index=col, columns=pd.to_datetime(dff[self.time_col]).dt.strftime('%Y-%m'), values=self.target, aggfunc=agg_func),2)

        stability = []
        for rating in ratings:
            stability.append(np.std(dff.loc[rating, :]))

        fig = go.Figure()
        plotly_main_layout(fig, title=f'Crop Stability | E[std(y)] = {round(np.mean(stability),2)}', x='Date', y=col, width=width, height=height, **kwargs)

        for rating, color in zip(ratings, colors):
            custom_data_values = dff.loc[rating, dff.columns].fillna(0).to_numpy() 
            if (stackgroup is not None) and (stackgroup is not False):
                fig.add_trace(go.Scatter(
                    x=dff.columns,
                    y=dff.loc[rating].values,  
                    marker=dict(color=color, size=8),
                    name=str(rating),
                    line=dict(width=3),
                    stackgroup=stackgroup,
                    mode='lines+markers'
                ))     
            else:         
                fig.add_trace(go.Scatter(
                    x=dff.columns,
                    y=dff.loc[rating].values,  
                    marker=dict(color=color, size=8),
                    name=str(rating),
                    line=dict(width=3),
                ))
            fig.update_traces(
                patch={
                    'customdata': custom_data_values,
                    'hovertemplate': 'Month: %{x}<br>Over: %{y}%<br>Volume: %{customdata}<extra></extra>'
                },
                selector=dict(name=str(rating)))

        return fig
    

    def plot_migration_analysis(self, df:pd.DataFrame=None, index:str='rating', column:str='rating', agg_func:str='count', 
                                z_normalizer:int=None, z_format:str=None, replace_0_None:bool=False,
                                initial_date:str=None, upto_date:str=None, sample:float=None, width=800, height=600,
                                show_fig:bool=True, colorscale:str='algae', xaxis_side:str='bottom', 
                                split:list|Literal['train','test','oot']=['train','test','oot']):
        '''
        Analyzes migration patterns within a dataset by aggregating values based on the given parameters. 
        The function generates a heatmap visualization of migration trends based on rating changes over time.
        
        ## Parameters

        - ```index``` : str, default='rating'
            Column name to be used as the index (rows) in the pivot table.
        
        - ```column``` : str, default='rating'
            Column name to be used as the columns in the pivot table.
        
        - ```agg_func``` : str, default='count'
            Aggregation function to be applied when summarizing data. Examples: 'sum', 'mean', 'count'.
        
        - ```z_normalizer``` : int, optional
            A normalization factor to adjust the z-values in the heatmap.
        
        - ```initial_date``` : str, optional
            The starting date (YYYY-MM-DD) for filtering the data.
        
        - ```upto_date``` : str, optional
            The ending date (YYYY-MM-DD) for filtering the data.
        
        - ```show_fig``` : bool, default=True
            If True, displays the generated heatmap.
        
        - ```colorscale``` : str, default='algae'
            Color scheme for the heatmap. Recommended options: 'algae', 'dense', 'amp'.
        
        - ```xaxis_side``` : str, default='bottom'
            Defines the placement of the x-axis labels ('top' or 'bottom').
        
        - ```z_format``` : str, optional
            Format specifier for the z-values in the heatmap.
        
        ## Returns:

        If `show_fig` is True:
            - Displays a heatmap visualization.
        If `show_fig` is False:
            - Returns a pivot table summarizing migration trends.

        '''
        if (df is not None):
            dff = df.copy(deep=True)
        else:
            dff = self.df.copy(deep=True)

        if isinstance(split, str):
            split = [split]
        try:
            dff = dff[dff['split'].isin(split)]
        except:
            if ('split' not in dff.columns) and (not getattr(self, 'suppress_warnings', False)):
                warnings.warn(
                    'No column ["split"] was found, therefore, the whole `df` will be used in this method',
                    category=UserWarning
                )
            dff = dff.copy(deep=True)
            
        if (sample is not None):
            sample = np.abs(sample)
            if (sample <= 1):
                dff = dff.sample(frac=sample)
            elif (sample > 1):
                if (sample > len(dff)):
                    raise ValueError(f"Sample size {sample} is larger than the dataset size {len(dff)}.")
                dff = dff.sample(n=sample)
        
        if initial_date is not None:
            initial_date = pd.to_datetime(initial_date)
            dff = dff[dff[self.time_col] >= initial_date]

        if upto_date is not None:
            upto_date = pd.to_datetime(upto_date)
            dff = dff[dff[self.time_col] <= upto_date]

        if (column == index):
            dff[f'{column}_'] = dff[column]
            index = f'{column}_'

        migration_dff = dff.groupby([index, column], observed=False)[self.target].agg(func=agg_func).reset_index().pivot(
            columns=column, index=index, values=self.target
        )
        if replace_0_None:
            migration_dff = migration_dff.replace(0, np.nan)

        if z_normalizer is None:
            z = list(reversed(migration_dff.values.tolist()))
            texttemplate = "%{z:.2f}"
        elif z_normalizer == 0:
            migration_dff = migration_dff.div(migration_dff.sum(axis=1), axis=0)
            z = list(reversed(migration_dff.values.tolist()))
            texttemplate = "%{z:.2f}"
        elif z_normalizer == 1:
            migration_dff = migration_dff.div(migration_dff.sum(axis=0), axis=1)        
            z = list(reversed(migration_dff.values.tolist()))
            texttemplate = "%{z:.2f}"

        if z_format == 'percent':
            z = [[elem * 100 for elem in sublist] for sublist in z]
            migration_dff = (100*migration_dff.round(4)).fillna('-')
            texttemplate = "%{z:.2f}"
        elif z_format == 'int':
            migration_dff = migration_dff.fillna(-10000).astype(int).replace(-10000,'-')
            texttemplate = "%{z:.0f}"
        else:
            try: migration_dff = migration_dff.round(3).fillna('-')
            except: migration_dff = migration_dff.fillna(-10000).astype(int).replace(-10000,'-')

        fig = go.Figure()
        fig.add_trace(go.Heatmap(
                z=z,
                x=migration_dff.columns,
                y=list(reversed(migration_dff.index)),
                hoverongaps=False,
                colorscale=colorscale,
                texttemplate=texttemplate
        ))
        plotly_main_layout(
            fig, title='Migration', x=column, y=index, width=width, height=height,
        )
        fig.update_layout({'xaxis':dict(gridcolor='#EEE',side=xaxis_side), 'yaxis':dict(gridcolor='#EEE')})
        
        if show_fig: return fig
        else: return migration_dff

    
    def plot_gains_per_risk_group(self, df:pd.DataFrame=None, initial_date:str=None, upto_date:str=None, col:str='rating',
                                  agg_func:str='mean', color_seq:px.colors=px.colors.sequential.Turbo, sample:float=None,
                                  show_bar:bool=True, show_scatter:bool=True, sort_by_bad:bool=False, 
                                  split:list|Literal['train','test','oot']=['train','test','oot'], width=800, height=600, **kwargs):
        
        if (df is not None):
            dff = df.copy(deep=True)
        else:
            dff = self.df.copy(deep=True)

        if isinstance(split, str):
            split = [split]
        try:
            dff = dff[dff['split'].isin(split)]
        except:
            if ('split' not in dff.columns) and (not getattr(self, 'suppress_warnings', False)):
                warnings.warn(
                    'No column ["split"] was found, therefore, the whole `df` will be used in this method',
                    category=UserWarning
                )
            dff = dff.copy(deep=True)
            
        if (sample is not None):
            sample = np.abs(sample)
            if (sample <= 1):
                dff = dff.sample(frac=sample)
            elif (sample > 1):
                if (sample > len(dff)):
                    raise ValueError(f"Sample size {sample} is larger than the dataset size {len(dff)}.")
                dff = dff.sample(n=sample)

        if initial_date is not None:
            initial_date = pd.to_datetime(initial_date)
            dff = dff[dff[self.time_col] >= initial_date]

        if upto_date is not None:
            upto_date = pd.to_datetime(upto_date)
            dff = dff[dff[self.time_col] <= upto_date]

        try: ratings = list(reversed(sorted(dff[col].unique())))
        except: raise TypeError('``rating`` column might have nan elements (not supported)')
        sample_points = [i / (len(ratings) - 1) for i in range(len(ratings))]
        colors = list(reversed(plotly.colors.sample_colorscale(color_seq, sample_points)))

        bad_percent_list = []; total_percent_list = []; total_list = []
        for rating in ratings:
            bad = dff[(dff[col] == rating)][self.target].agg(func=agg_func)
            bad_percent_list.append(100*bad)

            total = dff[col][(dff[col] == rating)].count()
            total_list.append(total)
            total_percent = round(100*total/len(dff),2)
            total_percent_list.append(total_percent)

        df = pd.DataFrame({'ratings': ratings, 'colors': colors, 'percent':bad_percent_list, 'total':total_list, 'percent total':total_percent_list})
        if sort_by_bad:
            colors = df['colors'].copy()
            df = df.sort_values(by='percent', ascending=False).reset_index(drop=True)
            del df['colors']
            df = pd.concat([df, colors], axis=1)
            df = df.astype({'ratings':str})

        fig = go.Figure()
        plotly_main_layout(fig, title='Gains per Risk Group', x=col, y='Percent', width=width, height=height, **kwargs)

        if show_scatter:
            fig.add_trace(trace=go.Scatter(
                x=df['ratings'], y=df['percent'],
                line=dict(color='#AAAAAA', width=3), mode='lines', showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=df['ratings'], y=df['percent'], mode='markers',
                marker=dict(color=df['colors'], size=10), name=str(rating), showlegend=False
            ))

        if show_bar:
            fig.add_trace(go.Bar(
                x=df['ratings'], y=df['percent total'],  
                marker=dict(color=df['colors']), name=str(rating),
                text=df['total'], showlegend=False
            ))
        return fig
    

    def plot_rating_infos(self, split:list|Literal['train','test','oot']=['train','test','oot'], width=1400, height=1000):
        if isinstance(split, str):
            split = [split]
        try:
            dff = self.df[self.df['split'].isin(split)].copy(deep=True)
        except:
            if ('split' not in self.df.columns) and (not getattr(self, 'suppress_warnings', False)):
                warnings.warn(
                    'No column ["split"] was found, therefore, the whole `df` will be used in this method',
                    category=UserWarning
                )
            dff = self.df.copy(deep=True)
        
        fig_gains = self.plot_gains_per_risk_group(split=split)['data']
        fig_stab = self.plot_stability_in_time(split=split)['data']
        fig_stab_ = self.plot_stability_in_time(split=split, agg_func='count', percent=False, stackgroup=True)['data']
        fig_ks = KS_Discriminant(dff, target=self.target, features='rating').plot()['data']

        fig = plotly.subplots.make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Gains por Risk Group', 
                'Kolmogorov-Smirnov Discriminant',
                f'Stability in Time | Metric',
                'Stability in Time | Volumetry',
            ),
            vertical_spacing=0.1,
            horizontal_spacing=0.05,
            y_title='percent [%] | volume [*]', 
        )
        for fig_ in fig_gains:
            fig.add_trace(fig_, row=1, col=1)
        for fig_ in fig_ks:
            fig.add_trace(fig_, row=1, col=2)
        for fig_ in fig_stab:
            fig.add_trace(fig_, row=2, col=1)
        for fig_ in fig_stab_:
            fig.add_trace(fig_, row=2, col=2)

        plotly_main_subplot_layout(fig, title=f'Basic Rating Evaluation | {self.target}', width=width, height=height)

        return fig