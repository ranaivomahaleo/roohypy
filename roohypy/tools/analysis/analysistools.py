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

def compute_log_return(data_item, return_lag=1):
    """
    This function returns the log-return of data (positive or null data).
    data is a dict that has the following structure:
        data['state_evolution'] : the evolution
        data['t'] : time
        data['label'] : data name
    
    Only not null data are considered thus we remove null values
    and adjust the scale of t.
    """
    
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
    
    return log_return, abs_log_return, t_log_return


def compute_autocorrelation_function(series, nlags=200):
    """
    This function returns the unbiased sample autocorrelation function.
    """
    autocorrelation = st.acf(series, unbiased=True, nlags=nlags)
    xlags = range(0, nlags, 1)
    
    return autocorrelation, xlags


