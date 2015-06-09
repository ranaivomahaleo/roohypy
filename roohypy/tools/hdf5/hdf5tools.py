# !/usr/bin/python
# -*- coding=utf-8 -*-
# Python 2.7 script (bitshuffle uses python 2)
#    Copyright (C) 2015 by
#    Ranaivo Razakanirina <ranaivo.razakanirina@atety.com>
#    All rights reserved.
#    BSD license.

from __future__ import division
import time
import sys
import os.path
import numpy as np
import roohypy.tools as tl
import roohypy.models as md
import scipy.sparse as sparse
import h5py as hdf
from h5py import h5f, h5d, h5z, h5t, h5s, filters
from bitshuffle import h5


def createGTHdf5File(datasetfullpath,
        shape=(100, 100, 100),
        epochs=20000,
        chunk_epoch=[[0, 1, 2], [3, 4, 5]],
        ordered_tuple_alpha_mu=[(20,20)],
        agents_id_list=[0, 1, 2],
        attributes={'title': ''},
        simulation={'simulationtitle': ''},
        network={'networkfolder': ''},
        chunksize=(100, 100, 100)):
    """This function inits and create an hdf5 file
    that have the following shape:
        (n_agents, alpha_mu, epochs)
    
    To accelerate the read process, the following 
    chunk size is considered:
        (n_agents, 100, 100)
        
    Returns:
    -------
    f : hdf5 file handler
        The reference to the hdf5 file.
    
    """
    f = hdf.File(datasetfullpath, 'w')
    
    for attribute in attributes:
        f.attrs[attribute] = attributes[attribute]
        
    for simulation_item in simulation:
        f.attrs[simulation_item] = simulation[simulation_item]
        
    for network_item in network:
        f.attrs[network_item] = network[network_item]
    
    # If select chunk=False (default value), 
    # save all data to hdf5 file
    # otherwise, select the configured part.
    if not simulation['selectchunk']:
        epochs_list = range(0, epochs, 1)
        validchunkidsarray = range(0, len(chunk_epoch), 1)
        f.create_dataset('t', data=epochs_list)
    else:
        epochs_list, \
        validchunkidsarray = \
        tl.getEpochsFromChunkIds(simulation['saved_chunkids'],
                                    chunk_epoch)
        f.create_dataset('t', data=epochs_list)
    
    f.attrs['selectchunk'] = simulation['selectchunk']
    f.create_dataset('validchunkidsarray', data=validchunkidsarray)
    
    f.create_dataset('alphas_mus', data=ordered_tuple_alpha_mu)
    f.create_dataset('agents_id', data=agents_id_list)

    filter_pipeline = (32008, 32000)
    filter_opts = ((64000000, h5.H5_COMPRESS_LZ4), ())
    
    for type in ['cash', 'goods', 'price']:
        h5.create_dataset(f,
                        type,
                        (shape[0], shape[1], shape[2]),
                        np.uint32,
                        chunks=chunksize,
                        filter_pipeline=filter_pipeline,
                        filter_opts=filter_opts)
        
        f[type].dims[0].label = 'agents_id'
        f[type].dims[1].label = 'alphas_mus'
        f[type].dims[2].label = 't'
        
        f[type].dims.create_scale(f['agents_id'])
        f[type].dims.create_scale(f['alphas_mus'])
        f[type].dims.create_scale(f['t'])
        
        f[type].dims[0].attach_scale(f['agents_id'])
        f[type].dims[1].attach_scale(f['alphas_mus'])
        f[type].dims[2].attach_scale(f['t'])

    return f
    
    
def loadGTIterationToHdf5File(f, pair_am, pair_t,
        cash, goods, price, integer_sensitivity=1,
        tuple_t_to_index={(0, 99): 0}):
    """This function loads each chunk data to the hdf5 file handler f.
    The data from cash, goods and price chunks are float data.
    Float data are transformed to integer data using the integer_sensitivity
    value.
    
    Returns
    -------
     f : hdf5 file handler
        The reference to the hdf5 file after loading the chunk data.
    
    """
    if tuple_t_to_index[pair_t] in f['validchunkidsarray']:
        txtinfo = ''
        txtinfo += '\n'
        txtinfo += 'Save' + str(pair_t) + ' chunk \n'
        txtinfo += 'chunk id=' + str(tuple_t_to_index[pair_t]) + ' chunk \n'
        print(txtinfo)

        c = np.round(np.dot(
            cash[:, 0:pair_am[1]-pair_am[0]+1, 0:pair_t[1]-pair_t[0]+1],
            integer_sensitivity), 0)
        g = np.round(np.dot(
            goods[:, 0:pair_am[1]-pair_am[0]+1, 0:pair_t[1]-pair_t[0]+1],
            integer_sensitivity), 0)
        p = np.round(np.dot(
            price[:, 0:pair_am[1]-pair_am[0]+1, 0:pair_t[1]-pair_t[0]+1],
            integer_sensitivity), 0)

        f['cash'][:, pair_am[0]:pair_am[1]+1, pair_t[0]:pair_t[1]+1] = c
        f['goods'][:, pair_am[0]:pair_am[1]+1, pair_t[0]:pair_t[1]+1] = g
        f['price'][:, pair_am[0]:pair_am[1]+1, pair_t[0]:pair_t[1]+1] = p
    
    return f
