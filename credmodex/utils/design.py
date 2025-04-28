from typing import List, Union, Optional
import plotly.graph_objects as go
import matplotlib.pyplot as plt




def plotly_main_layout(fig: go.Figure, width=700, height=600, x=None, y=None, title=None,
                       x_range=None, y_range=None, paper_color='white', 
                       customdata: Union[str, None] = None, hover_customdata='Info', 
                       hover_x='x', hover_y='y', **kwargs) -> go.Figure:
    fig.layout = go.Layout(
        width=width,
        height=height,
        plot_bgcolor=paper_color,
        paper_bgcolor=paper_color,
        xaxis={'gridcolor': '#cccccc', 'linecolor': 'black', 'title': x, 'range': x_range},
        yaxis={'gridcolor': '#cccccc', 'linecolor': 'black', 'title': y, 'range': y_range},
        title={'text': title},
    )
    
    # Now filter the kwargs to include only valid layout properties for Plotly
    valid_layout_keys = [
        'title', 'xaxis', 'yaxis', 'width', 'height', 'plot_bgcolor', 'paper_bgcolor', 'xaxis.title', 
        'yaxis.title', 'xaxis.range', 'yaxis.range', 'xaxis.gridcolor', 'yaxis.gridcolor'
    ]

    # Update the layout using valid kwargs
    fig.update_layout(
        **kwargs
    )
    
    if isinstance(customdata, str) and customdata == 'no':
        ...
    elif customdata is None:
        fig.update_traces(patch={
            'customdata': customdata, 'hovertemplate': hover_x + ': %{x}<br>' + hover_y + ': %{y}'
        })
    else:
        fig.update_traces(patch={
            'customdata': customdata,
            'hovertemplate': hover_x + ': %{x}<br>' + hover_y + ': %{y}<br>' + hover_customdata + ': %{customdata}<br>'
        })
    
    return fig
# ====================================================================================================
def plotly_main_subplot_layout(fig:go.Figure, width=1400, height=500, title=None, paper_color='white',
                        x=None, y=None, rows=1, cols=2, x_range=None, y_range=None,
                        customdata:Union[str, None]=None, hover_customdata='Info', 
                        hover_x='x',hover_y='y', **kwargs) -> go.Figure:
    fig.update_layout({
        'width':width,
        'height':height,
        'plot_bgcolor':paper_color,
        'paper_bgcolor':paper_color,
        'title':title,
        **kwargs
    })
    for xaxis in fig.select_xaxes():
        xaxis.update(
            showgrid=True,
            gridcolor='#CCCCCC',
            linecolor='black',
            title=x,
            range=x_range
        )
    for yaxis in fig.select_yaxes():
        yaxis.update(
            showgrid=True,
            gridcolor='#CCCCCC',
            linecolor='black',
            title=y,
            range=y_range
        )
    if isinstance(customdata, str) and customdata == 'no':
        ...
    elif customdata is None:
        fig.update_traces(patch={
            'customdata':customdata, 'hovertemplate': hover_x + ': %{x}<br>' + hover_y + ': %{y}'
        })
    else:
        fig.update_traces(patch={
            'customdata':customdata,
            'hovertemplate': hover_x + ': %{x}<br>' + hover_y + ': %{y}<br>' + hover_customdata + ': %{customdata}<br>'
        })
    return fig







def matplotlib_main_layout(fig, ax, width=7, height=6, x=None, y=None, title=None,
                           x_range=None, y_range=None, paper_color='white', **kwargs):
    # Set figure size
    fig.set_size_inches(width / 100, height / 100)
    
    # Set background color
    fig.patch.set_facecolor(paper_color)
    ax.set_facecolor(paper_color)
    
    # Set title
    if title is not None:
        ax.set_title(title)
    
    # Set x and y labels
    if x is not None:
        ax.set_xlabel(x)
    if y is not None:
        ax.set_ylabel(y)
    
    # Set x and y ranges
    if x_range is not None:
        ax.set_xlim(x_range)
    if y_range is not None:
        ax.set_ylim(y_range)
    
    # Set grid
    ax.grid(True, color='#cccccc')
    
    # Set axis line color
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    
    # Apply additional kwargs
    ax.tick_params(axis='both', which='both', length=0)  # Disable tick marks
    for key, value in kwargs.items():
        if key == 'gridcolor':
            ax.grid(True, color=value)
        elif key == 'title':
            ax.set_title(value)
        elif key == 'xaxis':
            if 'title' in value:
                ax.set_xlabel(value['title'])
            if 'range' in value:
                ax.set_xlim(value['range'])
        elif key == 'yaxis':
            if 'title' in value:
                ax.set_ylabel(value['title'])
            if 'range' in value:
                ax.set_ylim(value['range'])
    
    return fig, ax
