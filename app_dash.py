#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 13:18:31 2022

@author: aurelien
"""
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import dash_core_components as dcc
import dash_html_components as html
from dash import Dash
from dash.dependencies import Output, Input

def color_vec(df, i):
    vec = np.repeat('blue', len(df))
    vec[i] = 'red'
    return(vec)

with open('df.pickle', 'rb') as handle:
    df = pickle.load(handle)

df['Err'] = list(map(lambda x: np.abs(df['true'][x].squeeze() - df['pred'][x]), range(len(df))))

app = Dash(__name__)

app.layout = html.Div([
    html.H1(children="Absolute error analysis"),
    html.Div([
        html.Div([
            dcc.Dropdown(id='bathy',
                         options=[{'label': i, 'value': i} for i in df['bathy'].unique()],
                         value='2017-03-27'),
            dcc.Slider(id='slider_i',
                       value=0,
                       min=0,
                       tooltip={"placement": "bottom", "always_visible": True})
           ],style={'width': '80%', 'display': 'inline-block', 'align-items': 'center', 'justify-content': 'center'})
        ]),
        html.Div(dcc.Graph(id='plotly'),style={'width': '100%', 'display': 'inline-block', 'align-items': 'center', 'justify-content': 'center'})
    ])

@app.callback(
    Output('slider_i', component_property='max'),
    Input('bathy', 'value')
)
def update_slider(bathy):
    dff = df[df['bathy'] == str(bathy)].reset_index()
    return len(dff)-1

@app.callback(
    Output('plotly', 'figure'),
    Input('bathy', 'value'),
    Input('slider_i', 'value'),)
def update_graph(bathy, index):
    dff = df[df['bathy'] == str(bathy)].reset_index()
    
    i = int(index)

    fig = make_subplots(2, 3)
    _vmin, _vmax = np.min(dff['true'][i])-1, np.max(dff['true'][i])+1

    fig.add_trace(
        go.Heatmap(z=dff['input'][i][:,:,0], colorscale='gray', showscale=False), row=1, col=1)

    fig.add_trace(
        go.Heatmap(z=dff['input'][i][:,:,1], colorscale='gray', showscale=False), row=1, col=2)

    fig.add_trace(
        go.Scatter(mode='markers', x=dff['Date'], y=dff['Tide'], marker=dict(size=10, color=color_vec(dff,i))), row=1, col=3)

    fig.add_trace(
        go.Heatmap(z=dff['true'][i].squeeze(), colorscale='jet', colorbar=dict(x=0.29, y=0.21, len=.45), zmin=_vmin, zmax=_vmax), row=2, col=1)

    fig.add_trace(
        go.Heatmap(z=dff['pred'][i], colorscale='jet', colorbar=dict(x=0.645, y=0.21, len=.45) , zmin=_vmin, zmax=_vmax), row=2, col=2)

    fig.add_trace(
        go.Heatmap(z=dff['Err'][i], colorscale='inferno', colorbar=dict(x=1, y=0.21, len=.45)), row=2, col=3)
    fig.update_layout(plot_bgcolor = "white", transition_duration=10)
    fig.update_layout(
    autosize=True,
    height=1000)
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
