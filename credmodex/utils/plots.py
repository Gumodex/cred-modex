import pandas as pd
import numpy as np
from typing import Literal
import warnings

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .design import plotly_main_layout, plotly_main_subplot_layout





def plot_migration_analysis(df:pd.DataFrame=None, index:str='rating', column:str='rating', target:str=None, 
                            agg_func:str='count', z_normalizer:int=None, z_format:str=None, replace_0_None:bool=False, 
                            time_col:str=None, initial_date:str=None, upto_date:str=None, sample:float=None, 
                            width=800, height=600, show_fig:bool=True, colorscale:str='algae', xaxis_side:str='bottom', 
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
        raise ValueError(f"No dataframe defined!")

    if (target is None):
        raise ValueError(f"No target defined")

    if isinstance(split, str):
        split = [split]
    try:
        dff = dff[dff['split'].isin(split)]
    except:
        warnings.warn("No split column found with 'train', 'test', or 'oot' elements.")
        
    if (sample is not None):
        sample = np.abs(sample)
        if (sample <= 1):
            dff = dff.sample(frac=sample)
        elif (sample > 1):
            if (sample > len(dff)):
                raise ValueError(f"Sample size {sample} is larger than the dataset size {len(dff)}.")
            dff = dff.sample(n=sample)
    
    if (initial_date is not None):
        try:
            initial_date = pd.to_datetime(initial_date)
            dff = dff[dff[time_col] >= initial_date]
        except Exception as e:
            raise ValueError(f"Failed to parse initial_date or filter DataFrame: {e}")

    if (upto_date is not None):
        try:
            upto_date = pd.to_datetime(upto_date)
            dff = dff[dff[time_col] <= upto_date]
        except Exception as e:
            raise ValueError(f"Failed to parse initial_date or filter DataFrame: {e}")

    if (column == index):
        dff[f'{column}_'] = dff[column]
        index = f'{column}_'

    migration_dff = dff.groupby([index, column], observed=False)[target].agg(func=agg_func).reset_index().pivot(
        columns=column, index=index, values=target
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
        migration_dff = (100*migration_dff.round(4))
        texttemplate = "%{z:.2f}"
    elif z_format == 'int':
        migration_dff = migration_dff.astype('Int64')
        texttemplate = "%{z:.0f}"
    else:
        try: migration_dff = migration_dff.round(3)
        except: migration_dff = migration_dff.astype('Int64')

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




def subplot_migration_analysis(df:pd.DataFrame=None, index:str='rating', column:str='rating', target:str=None, 
                               agg_func:list[str]=['mean','count'], z_normalizer:list[int]=[None,None], z_format:list[str]=['percent','int'], 
                               replace_0_None:list[bool]=[False,False], time_col:str=None, initial_date:str=None, 
                               upto_date:str=None, sample:float=None, width=1400, height=600, colorscale:list[str]=['amp','algae'],
                               split:list|Literal['train','test','oot']=['train','test','oot']):

    if (df is not None):
        dff = df.copy(deep=True)
    else:
        raise ValueError(f"No dataframe defined!")

    if (target is None):
        raise ValueError(f"No target defined")

    fig = make_subplots(rows=1, cols=2, subplot_titles=agg_func)

    trace = plot_migration_analysis(
        df=df, target=target, split=split, time_col=time_col, initial_date=initial_date, upto_date=upto_date,
        z_normalizer=z_normalizer[0], z_format=z_format[0], replace_0_None=replace_0_None[0], sample=sample,
        column=column, index=index, agg_func=agg_func[0], colorscale=colorscale[0]
    )['data']
    for t in trace:
        t.colorbar = dict(x=0.45)
        fig.add_trace(t, row=1, col=1)

    trace = plot_migration_analysis(
        df=df, target=target, split=split, time_col=time_col, initial_date=initial_date, upto_date=upto_date,
        z_normalizer=z_normalizer[1], z_format=z_format[1], replace_0_None=replace_0_None[1], sample=sample,
        column=column, index=index, agg_func=agg_func[1], colorscale=colorscale[1]
    )['data']
    for t in trace:
        t.colorbar = dict(x=1)
        fig.add_trace(t, row=1, col=2)

    plotly_main_subplot_layout(fig, 
        title=f'Migration Analysis | (metric: {target})',
        x=column, y=index, width=width, height=height
    )

    return fig