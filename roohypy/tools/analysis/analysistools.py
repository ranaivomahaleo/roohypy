# !/usr/bin/python
# -*- coding=utf-8 -*-
# Python 2.7 script (bitshuffle uses python 2)
#    Copyright (C) 2015 by
#    Ranaivo Razakanirina <ranaivo.razakanirina@atety.com>
#    All rights reserved.
#    BSD license.

from __future__ import division

import numpy as np
import statsmodels.tsa.stattools as st
import scipy.stats as scst
import scipy.signal as signal
from pypr.stattest.ljungbox import *

def compute_log_return(data_item, return_lag=1, **args):
    """
    This function returns the log-return of data (positive or null data).
    data is a dict that has the following structure:
        data['state_evolution'] : the evolution
        data['t'] : time
        data['label'] : data name
    
    Only not null data are considered thus we remove null values
    and adjust the scale of t.
    """
    return_lag = int(return_lag)
    
    # Filter data and t that correspond to positive values of data
    cond_index = np.where(data_item['state_evolution'] > 0)
    data = data_item['state_evolution'][cond_index]
    t = data_item['t'][cond_index]

    # Compute log-return
    rshift_state_evolution = np.roll(data, return_lag)
    r = np.log10(data / rshift_state_evolution)
    log_return = r[-(r.size-return_lag)::]
    t_log_return = t[-(r.size-return_lag)::]
    
    # Compute abs log-return
    abs_log_return = np.abs(log_return)
    
    # Check if sign_stats argument is set.
    # If yes, use only the last sign_stats values
    if 'sign_stats' in args.keys():
        sign_stats = int(args['sign_stats'])
        log_return = log_return[-sign_stats::]
        t_log_return = t_log_return[-sign_stats::]
        abs_log_return = abs_log_return[-sign_stats::]
        y_filtered = data[-sign_stats::]
        t_filtered = t[-sign_stats::]
    
    return log_return, abs_log_return, t_log_return, y_filtered, t_filtered


def compute_distribution_norm_fit(series, bins=100, density=True, serieslabel=''):
    """
    This function computes the distribution of series.
    """
    bins = int(bins)
    
    hist_series = np.histogram(series, bins=bins, density=density)
    x = hist_series[1]
    y = hist_series[0]
    
    
    # Fit to normal distribution
    (mu, sigma) = scst.norm.fit(series)
    
    txtinfo = 'Series label: ' + serieslabel + '\n'
    txtinfo += 'Fit with normal' + '\n'
    txtinfo += 'mu= ' + str(mu) + '\n'
    txtinfo += 'sigma= ' + str(sigma) + '\n'
    print(txtinfo)
    
    xnorm = np.linspace(min(x), max(x), bins)
    ynorm = scst.norm.pdf(xnorm, loc=mu, scale=sigma)
    
    return x, y, xnorm, ynorm


def compute_autocorrelation_function(series, nlags=200):
    """
    This function returns the unbiased sample autocorrelation function.
    """
    autocorrelation = st.acf(series, unbiased=True, nlags=nlags)
    xlags = range(0, nlags, 1)
    
    return autocorrelation, xlags


def compute_lbtest(series, lags=20, alpha=0.05, serieslabel=''):
    """
    Compute LB test for null values of sample autocorrelation function.
    """
    alpha = float(alpha)
    
    txtinfo = 'Series label: ' + serieslabel + '\n'
    print(txtinfo)
    
    h, pV, Q, cV = lbqtest(series, range(1, lags), alpha=alpha)
    print 'lag   p-value          Q    c-value   rejectH0'
    for i in range(len(h)):
        print "%-2d %10.3f %10.3f %10.3f      %s" % (i+1, pV[i], Q[i], cV[i], str(h[i]))


def compute_adf_test(series, serieslabel=''):
    """
    Compute ADF test for stationarity.
    """
    txtinfo = 'Series label: ' + serieslabel + '\n'
    adf = st.adfuller(series, 1)
    txtinfo += str(adf) + '\n'
    print(txtinfo)


def compute_periodogram(series, serieslabel=''):
    """
    This function compute the frequential representation of series.
    """
    txtinfo = 'Series label: ' + serieslabel + '\n'
    print(txtinfo)
    
    period = signal.periodogram(series)
    xfreq = period[0]
    yfreq = period[1]

    return xfreq, yfreq

