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
import roohypy.tools.hdf5 as hd
import scipy.sparse as sparse


def InitGTSimulation(simulation, network, attributes={}, simulation_index=0):
    """This function inits all necessary temporary variables
    needed for a GT simulation.
    """
    # Process network data
    nodefilepath = network['networkfolder'] + network['networkname']  + '/nodes.csv'
    edgefilepath = network['networkfolder'] + network['networkname']  + '/edges.csv'
    A = md.edgeNodeCsvToAdj(nodefilepath, edgefilepath)
    n, csc_A, elt_indices, elt_indices_tr, attributes = md.getMainNetworkCharacteristics(A)
    attributes['networkname'] = network['networkname']
    
    # Process the initial conditions
    # and some attributes of the simulation
    if simulation['rand_ic']==True:
        c_ic, g_ic, p_ic = tl.getRandomUniformIC(
            c_tot=simulation['c_tot'],
            g_tot=simulation['g_tot'],
            alpha_mu_interval=simulation['alpha_mu_interval'],
            c_min_lim=simulation['c_min_lim'],
            g_min_lim=simulation['g_min_lim'],
            n=n
        )
    else:
        c_ic, g_ic, p_ic  = tl.getHomogeneousInitialConditions(
            simulation['c0'], 
            simulation['g0'], 
            simulation['p0'],
            n
        )

    # Get all possible combinations of values of alpha and mu
    # and build the chunks for epochs and alpha_mu
    alphas_mus, alphas_mus_indices, alpha_mu_to_index, index_to_alpha_mu = md.getListOfAlphaMu(simulation['alpha_mu_interval'])
    n_combinations = len(alphas_mus)
    
    chunk_alpha_mu = tl.chunkList(alphas_mus_indices, simulation['alpha_mu_chunk_size'])
    chunk_am = tl.chunkList(alphas_mus, simulation['alpha_mu_chunk_size'])
    chunk_epoch = tl.chunkList(range(0, simulation['epochs'], 1), simulation['epochs_chunk_size'])
    epoch_min = list(map(lambda x: np.min(x), chunk_epoch))
    epoch_max = list(map(lambda x: np.max(x), chunk_epoch))
    tuple_t = zip(epoch_min, epoch_max)
    am_min = list(map(lambda x: np.min(x), chunk_alpha_mu))
    am_max = list(map(lambda x: np.max(x), chunk_alpha_mu))
    tuple_am = zip(am_min, am_max)

    one = np.ones((n, simulation['alpha_mu_chunk_size'], simulation['epochs_chunk_size']))
    cashini = one * c_ic.reshape((n, 1, 1))
    goodsini = one * g_ic.reshape((n, 1, 1))
    priceini = one * p_ic.reshape((n, 1, 1))

    cash = np.zeros((n, simulation['alpha_mu_chunk_size'], simulation['epochs_chunk_size']+1))
    goods = np.zeros((n, simulation['alpha_mu_chunk_size'], simulation['epochs_chunk_size']+1))
    price = np.zeros((n, simulation['alpha_mu_chunk_size'], simulation['epochs_chunk_size']+1))
    
    # Build the temporary arrays and vectors for optimized GT-Model
    zeros, zeros1, zeros_vector, zeros_vector1, zeros_vector2, zeros_vector3 = md.getNullArraysAndVectors(n)
    
    # Create results folder and files
    resultname = tl.getResultFolderName(networkname=network['networkname'],
                                            step=simulation['alpha_mu_interval'],
                                            epochs=simulation['epochs'],
                                            integer_sensitivity=simulation['integer_sensitivity'])
    datasetfolder = simulation['resultfolder'] + resultname + '/'
    tl.checkCreateFolder(datasetfolder)
    datasetfullpath = datasetfolder + 'dataset_' + str(simulation_index) + '.h5'
    
    # Create hdf5 file
    f = hd.createGTHdf5File(datasetfullpath,
        shape=(n, n_combinations, simulation['epochs']),
        chunksize=(n, simulation['alpha_mu_chunk_size'], simulation['epochs_chunk_size']),
        epochs=simulation['epochs'],
        ordered_tuple_alpha_mu=alphas_mus,
        agents_id_list=range(0, n, 1),
        attributes=attributes,
        simulation=simulation,
        network=network
    )
    
    iterate = {}
    iterate['f'] = f
    iterate['tuple_am'] = tuple_am
    iterate['tuple_t'] = tuple_t
    iterate['cashini'] = cashini
    iterate['goodsini'] = goodsini
    iterate['priceini'] = priceini
    iterate['index_to_alpha_mu'] = index_to_alpha_mu
    iterate['A'] = A
    iterate['csc_A'] = csc_A
    iterate['elt_indices'] = elt_indices
    iterate['elt_indices_tr'] = elt_indices_tr
    iterate['zeros'] = zeros
    iterate['zeros1'] = zeros1
    iterate['zeros_vector'] = zeros_vector
    iterate['zeros_vector1'] = zeros_vector1
    iterate['zeros_vector2'] = zeros_vector2
    iterate['zeros_vector3'] = zeros_vector3

    return cash, goods, price, iterate


def LaunchGTSimulation(simulation, network, attributes={}, simulation_index=0):
    """This function launches a GT simulation.
    """
    # Init and launch the GT simulation
    cash, goods, price, iterate = InitGTSimulation(simulation, network, attributes=attributes, simulation_index=simulation_index)

    for pair_am in iterate['tuple_am']:
        cash[:,:,0] = iterate['cashini'][:,:,0]
        goods[:,:,0] = iterate['goodsini'][:,:,0]
        price[:,:,0] = iterate['priceini'][:,:,0]
        for pair_t in iterate['tuple_t']:
            print('Network ' + network['networkname'])
            print(pair_t)
            print(pair_am)
            start_time = time.time()

            # Compute each chunk
            cash, goods, price = md.optimizedGTModel6(
                pair_am, pair_t,
                iterate['index_to_alpha_mu'],
                iterate['A'], iterate['csc_A'], 
                iterate['elt_indices'], iterate['elt_indices_tr'],
                iterate['zeros'], iterate['zeros1'],
                iterate['zeros_vector'], iterate['zeros_vector1'],
                iterate['zeros_vector2'], iterate['zeros_vector3'],
                cash, goods, price)
            
            # Load in f
            iterate['f'] = hd.loadGTIterationToHdf5File(iterate['f'], pair_am, pair_t,
                cash, goods, price,
                integer_sensitivity=simulation['integer_sensitivity'])
        
            # Take the last evolution and put it as initial condition 
            # of the next chunk
            cash[:,:,0] = cash[:,:,pair_t[1]-pair_t[0]+1]
            goods[:,:,0] = goods[:,:,pair_t[1]-pair_t[0]+1]
            price[:,:,0] = price[:,:,pair_t[1]-pair_t[0]+1]
        
            #   print(cash)
        
            end_time = time.time()
            print(end_time - start_time)
            print('-----------')

    # Flush hdf5 memory to file
    iterate['f'].flush()
