# !/usr/bin/python
# -*- coding=utf-8 -*-
# Python 2 script (this choice is due to bitshuffle working under Python 2)

from __future__ import division

# https://pypi.python.org/pypi/plotly
import plotly.plotly as py
from plotly.graph_objs import *

def getDefaultPlotlyLayout(xtitle='', ytitle='',
        xautorange=False, yautorange=False,
        xrange=[0,1], yrange=[0,1],
        tickfontsize=23,
        r=10, b=80, t=10, l=140,
        xlegend=0.8, ylegend=0.9, legendsize=18
        ):
    """
    """
    layout = Layout(
        width=1000,
        height=618,
        margin=Margin(
            r=r,
            b=b,
            t=t,
            l=l
        ),
        xaxis=XAxis(
            title = xtitle,
            zeroline=False,
            showline=True,
            ticks='inside',
            ticklen=15,
            tickwidth=3,
            linewidth=2,
            mirror='ticks',
            autorange=xautorange,
            range=xrange,
            titlefont = Font(
                family='Open sans',
                size = 40
            ),
            tickfont = Font(
                family='Open sans',
                size = tickfontsize
            ),
        ),
        yaxis=YAxis(
            title = ytitle,
            zeroline=False,
            showline=True,
            ticks='inside',
            ticklen=15,
            tickwidth=3,
            linewidth=2,
            mirror='ticks',
            autorange=yautorange,
            range=yrange,
            titlefont = Font(
                family='Open sans',
                size = 40
            ),
            tickfont = Font(
                family='Open sans',
                size = tickfontsize
            ),
        ),
        legend=Legend(
            x=xlegend,
            y=ylegend,
            font = Font(
                family='Open sans',
                size = legendsize
            ),
        ),
        showlegend=True
    )
    return layout
    

def getDefaultPlotlyMarker(size=10):
    """
    """
    marker = Marker(
        size=size,
        color='rgb(0, 0, 0)'
    )
    return marker
    
def getDefaultPlotlyContours(**kwargs):
    """
    """
    return Contours(
        showlines = False,
        **kwargs
    )
    
def getDefaultPlotlyColorbar(colorbartickfont=28):
    """
    """
    return ColorBar(
                tickfont = Font(
                    family='Open sans',
                    size = colorbartickfont
                ),
            )
    
def getDefaultPlotlyParameterBasinLayout(xtitle='', ytitle='',
        xautorange=False, yautorange=False,
        xrange=[0,1], yrange=[0,1],
        tickfontsize=23,
        r=10, b=80, t=10, l=140
    ):
    """
    """
    layout = Layout(
        width=1033,
        height=897,
        margin=Margin(
            r=r,
            b=b,
            t=t,
            l=l
        ),
        xaxis=XAxis(
            title = xtitle,
            zeroline=False,
            showline=True,
            ticks='outside',
            ticklen=7,
            tickwidth=3,
            linewidth=2,
            mirror='ticks',
            autorange=xautorange,
            range=xrange,
            titlefont = Font(
                family='Open sans',
                size = 40
            ),
            tickfont = Font(
                family='Open sans',
                size = tickfontsize
            ),
        ),
        yaxis=YAxis(
            title = ytitle,
            zeroline=False,
            showline=True,
            ticks='outside',
            ticklen=7,
            tickwidth=3,
            linewidth=2,
            mirror='ticks',
            autorange=yautorange,
            range=yrange,
            titlefont = Font(
                family='Open sans',
                size = 40
            ),
            tickfont = Font(
                family='Open sans',
                size = tickfontsize
            ),
        ),
        legend=Legend(
            x=0.8,
            y=0.9,
            font = Font(
                family='Open sans',
                size = 18
            ),
        ),
        showlegend=False
    )
    return layout
    

def getDefaultPlotlyEvolutionLayout(xtitle='', ytitle='',
        xautorange=False, yautorange=False,
        xrange=[0,1], yrange=[0,1],
        tickfontsize=23,
        r=10, b=80, t=10, l=140):
    """
    """
    layout = Layout(
        width=1000,
        height=618,
        margin=Margin(
            r=r,
            b=b,
            t=t,
            l=l
        ),
        xaxis=XAxis(
            title = xtitle,
            zeroline=False,
            showline=True,
            ticks='inside',
            ticklen=15,
            tickwidth=3,
            linewidth=2,
            mirror='ticks',
            autorange=xautorange,
            range=xrange,
            exponentformat="none",
            titlefont = Font(
                family='Open sans',
                size = 40
            ),
            tickfont = Font(
                family='Open sans',
                size = tickfontsize
            ),
        ),
        yaxis=YAxis(
            title = ytitle,
            zeroline=False,
            showline=True,
            ticks='inside',
            ticklen=15,
            tickwidth=3,
            linewidth=2,
            mirror='ticks',
            autorange=yautorange,
            range=yrange,
            exponentformat="none",
            titlefont = Font(
                family='Open sans',
                size = 40
            ),
            tickfont = Font(
                family='Open sans',
                size = tickfontsize
            ),
        ),
        legend=Legend(
            x=1.03,
            y=0.9,
            font = Font(
                family='Open sans',
                size = 18
            ),
        ),
        showlegend=True
    )
    return layout


def buildPlotlyTrace(x, y):
    """
    plotly importation is mandatory
    This function builds the plot traces.
    
    ('x', array([    0,     1,     2, ..., 19997, 19998, 19999]))
    ('y', [(0, array([ 50000.       ,  49775.640625 ,  48062.203125 , ...,  47292.1015625,
        47292.1015625,  47292.1015625], dtype=float32)), (2, array([ 50000.       ,  61115.640625 ,  54594.9140625, ...,  68104.859375 ,
        68104.859375 ,  68104.859375 ], dtype=float32)), (3, array([ 50000.        ,  42622.86328125,  44647.65625   , ...,
        35597.96875   ,  35597.96875   ,  35597.96875   ], dtype=float32)), (4, array([ 50000.        ,  55166.66796875,  54519.6015625 , ...,
        64020.3125    ,  64020.3125    ,  64020.3125    ], dtype=float32))])
    """
    traces = list()
    for trace_name in y:
        traces.append(Scatter(x=x, y=trace_name[1], name=trace_name[0]))
    return traces