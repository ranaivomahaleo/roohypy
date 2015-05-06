# !/usr/bin/python
# -*- coding=utf-8 -*-
#    Copyright (C) 2015 by
#    Ranaivo Razakanirina <ranaivo.razakanirina@atety.com>
#    All rights reserved.
#    BSD license.

from __future__ import division

import roohypy as rp
import numpy as np
import json as json
import roohypy as rp
    
def getGTParameterBassinFromGTTransposedHdf5(hdf5GTdataset,
                                    last_epochs=100,
                                    type='cash',
                                    agent_id=0,
                                    sensitivity=10000):
    """
    Get parameter bassin from GT Transposed Hdf5 data set
    "Transposed" means that the shape of the hdf5 file
    is (agent_ids, alpha_mu, epochs)
    """
    # Manage networname name for some data sets
    if 'networkname' in hdf5GTdataset.attrs:
        networkname = hdf5GTdataset.attrs['networkname']
    else:
        networkname = hdf5GTdataset.attrs['networname']
    
    integer_sensitivity = hdf5GTdataset.attrs['integer_sensitivity']
    
    x_y = np.array(hdf5GTdataset[type].dims[1][0])
    data = hdf5GTdataset[type][agent_id,:,-last_epochs:]
    data = np.transpose(data) # Mandatory
    
    X = np.unique(x_y[:,0])
    Y = np.unique(x_y[:,1])
    data = np.transpose(data)
    K = list(map(lambda y: rp.tools.countuniquevalue(y, sensitivity=sensitivity), data))
    K = np.reshape(K, (len(X), len(Y)))
    K = np.transpose(K)
    return K, X, Y, networkname


def getGTBifurcationFromGTHdf5(hdf5GTdataset,
                        last_epochs=100,
                        type='cash',
                        agent_id=0,
                        xaxis='alpha',
                        fixed=200):
    """
    This function get bifurcation datas (x and y)
    """
    # Manage networname name for some data sets
    if 'networkname' in hdf5GTdataset.attrs:
        networkname = hdf5GTdataset.attrs['networkname']
    else:
        networkname = hdf5GTdataset.attrs['networname']

    title = {'cash': 'c_', 'goods': 'g_', 'price': 'p_'}
    ytitle = '$' + title[type] +'{'+ str(agent_id)+'}'+'$'
    
    alphas_mus = np.array(hdf5GTdataset[type].dims[1][0])
    if xaxis=='alpha':
        mu = fixed
        X = np.unique(alphas_mus[:,0])
        index_list = list(map(lambda y: rp.tools.getIndexOf2DNpArray(alphas_mus, y, mu), X))
        xtitle = '$\\alpha$'
        fixedlegend = '$\\mu=' + str(fixed/1000) + '$'
    if xaxis=='mu':
        alpha = fixed
        X = np.unique(alphas_mus[:,1])
        index_list = list(map(lambda y: rp.tools.getIndexOf2DNpArray(alphas_mus, alpha, y), X))
        xtitle = '$\\mu$'
        fixedlegend = '$\\alpha=' + str(fixed/1000) + '$'
    data = hdf5GTdataset[type][-last_epochs:, index_list, agent_id]
    data = np.transpose(data)

    datax = np.repeat(X, last_epochs)
    datay = np.reshape(data, (data.shape[0] * data.shape[1],))
    
    return datax, datay, fixedlegend, xtitle, ytitle, networkname
    

def getGTTimeEvolutionFromGTTransposedHdf5(hdf5GTdataset,
                        type='cash',
                        agent_ids=[0],
                        alpha=100,
                        mu=100):
    """
    This function returns the time evolution of list of agents
    from hdf5 files.
    "Transposed" means that the shape of the hdf5 file
    is (agent_ids, alpha_mu, epochs)
    """
    # Manage networname name for some data sets
    if 'networkname' in hdf5GTdataset.attrs:
        networkname = hdf5GTdataset.attrs['networkname']
    else:
        networkname = hdf5GTdataset.attrs['networname']
        
    integer_sensitivity = hdf5GTdataset.attrs['integer_sensitivity']
    
    datax = np.array(hdf5GTdataset[type].dims[2][0])
    alphas_mus = np.array(hdf5GTdataset[type].dims[1][0])
    am_index = rp.tools.getIndexOf2DNpArray(alphas_mus, alpha, mu)
    if agent_ids == 'all':
        agent_labels = np.array(hdf5GTdataset[type].dims[0][0])
        datay = hdf5GTdataset[type][:, am_index, :]
    else:
        agent_labels = sorted(agent_ids)
        datay = hdf5GTdataset[type][sorted(agent_ids), am_index, :]
    datay = np.transpose(datay) # Mandatory
    datay = np.dot(datay, 1/integer_sensitivity)
    max_y = np.max(datay)
    
    return datax, zip(agent_labels, np.transpose(datay)), networkname, max_y
