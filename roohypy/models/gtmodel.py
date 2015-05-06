# !/usr/bin/python
# -*- coding=utf-8 -*-
# Python 3

#    Copyright (C) 2015 by
#    Ranaivo Razakanirina <ranaivo.razakanirina@atety.com>
#    All rights reserved.
#    BSD license.

from __future__ import division

import networkx as nx
import numpy as np
import numpy.linalg as la
import roohypy.core as cr
import roohypy.tools as tl
import time
import scipy.sparse as sparse
import scipy.weave as weave


def getListOfAlphaMu(step_parameter_interval):
    """
    """
    mus = range(step_parameter_interval, 1000, step_parameter_interval)
    alphas = range(step_parameter_interval, 1000, step_parameter_interval)
    alphas_mus, alphas_mus_indices, alpha_mu_to_index, index_to_alpha_mu = tl.listAllOrderedCombinations(alphas, mus)
    return alphas_mus, alphas_mus_indices, alpha_mu_to_index, index_to_alpha_mu
    
    
def edgeNodeCsvToAdj(nodefilepath, edgefilepath):
    """
    Return the adjacency matrix in np.array format
    corresponding to the nodefilepath and edgefilepath csv files.
    """
    dgraph = nx.DiGraph()
    
    nodes = np.genfromtxt(nodefilepath, skip_header=1, dtype=np.uint32)
    edges = np.genfromtxt(edgefilepath, skip_header=1, dtype=np.uint32)
    
    dgraph.add_nodes_from(nodes[:, 1])
    dgraph.add_edges_from(edges[:, 1:])
    
    A = nx.to_numpy_matrix(dgraph, dtype=np.uint32)
    A = np.array(A)
    
    return A
    
    
def getMainNetworkCharacteristics(A):
    """
    This function returns:
    - n: 
    - csc_A: The sparse CSC format representation of the adjacency matrix A
    - 
    """
    csc_A = sparse.csc_matrix(A)
    n = len(A)
    
    nonzero_indices = csc_A.nonzero()
    elt_indices = (np.vstack((nonzero_indices[0],nonzero_indices[1]))).transpose()
    elt_indices_tr = (np.vstack((nonzero_indices[1],nonzero_indices[0]))).transpose()

    attributes = {}
    dgraph = nx.from_numpy_matrix(A, create_using=nx.DiGraph())
    m = nx.number_of_edges(dgraph)
    attributes['n'] = nx.number_of_nodes(dgraph)
    attributes['m'] = m
    attributes['p_rand'] = m / (n * (n-1))
    attributes['directed'] = True
    attributes['acyclic'] = nx.is_directed_acyclic_graph(dgraph)
    attributes['n_scc'], attributes['n_cc'] = cr.getconnectedcomponents(dgraph)    
    attributes['title'] = ''
    attributes['networname'] = ''
    attributes['p'] = 0
    attributes['m0'] = 0
    attributes['m_links'] = 0
    attributes['beta'] = 0
    attributes['k'] = 0
    attributes['algorithm'] = ''

    return n, csc_A, elt_indices, elt_indices_tr, attributes


def GTModel(c, g, p, A, alpha, mu):
    """
    This function returns the t+1 values of a GT model
    """
    n_rows = A.shape[0]
    one = np.ones(n_rows)

    C_f = np.dot(
        np.diag(alpha * c),
        tl.rows_sum_norm(
            np.dot(A, la.inv(np.diag(p)))
        )
    )

    G_f = np.dot(
        np.diag(mu*g),
        tl.rows_sum_norm(np.transpose(C_f))
    )

    c_next = (one-alpha) * c + np.dot(np.transpose(C_f), one)
    g_next = (one-mu) * g + np.dot(np.transpose(G_f), one)
    p_next = np.dot(
        la.inv(np.diag(mu*g)),
        np.dot(np.transpose(C_f), one)
    )

    return c_next, g_next, p_next


def c_compute_cash_flow(o, d, invp, indices):
    code = \
    """
    int m, i, j;
    for (m=0; m<Nindices[0]; m++) {
        i = INDICES2(m,0);
        j = INDICES2(m,1);
        O2(i,j) = D1(i) * INVP1(j);
    }
    """
    weave.inline(code, ['o', 'd', 'invp', 'indices'])


def c_compute_goods_flow(o, d, cftr, indices):
    code = \
    """
    int m, i, j;
    for (m=0; m<Nindices[0]; m++) {
        i = INDICES2(m,0);
        j = INDICES2(m,1);
        O2(i,j) = D1(i) * CFTR2(i,j);
    }
    """
    weave.inline(code, ['o', 'd', 'cftr', 'indices'])


def c_compute_price(o, cf, gf, indices):
    code = \
    """
    int m, i, j;
    for (m=0; m<Nindices[0]; m++) {
        i = INDICES2(m,0);
        j = INDICES2(m,1);
        O1(j) = CF2(i,j) / GF2(j,i);
    }
    """
    weave.inline(code, ['o', 'cf', 'gf', 'indices'])

    
def optimizedGTModel6(
        pair_am, pair_t,
        index_to_alpha_mu,
        A, csc_A, 
        elt_indices, elt_indices_tr,
        zeros, zeros1, zeros_vector, zeros_vector1, zeros_vector2, zeros_vector3,
        cash, goods, price):
    """
    This function is the optimized version (v6) of the GT-Model dynamics.
    The input are the cash, goods and price arrays with shape
    (:, chunk_alpha_mu, 1) and the function returns in batch the time evolution
    of cash, goods and price.
    
    The results corresponding to this optimized GT Model v6 update function
    are store inside an HDF5 file with the following shape:
    (agent_ids, alpha_mu, epochs).
    
    Parameters
    ----------
    pair_am : pair
        This pair consists of the start and end indices 
        of alpha, mu combinations.
        pair_am = (2000, 2099) means that we compute
        the state of each agent corresponding to the
        alpha,mu indices 2000 to 2099.
        
    pair_t : pair
        This pair consists of the start and end indices
        of epochs.
        pair_t = (100, 199) means that we compute
        the evolution from t=100 to t=199 on batch.
        
    index_to_alpha_mu : dict
        Conversion table from index to the real values of alpha and mu
        
    A, csc_A : numpy array, numpy sparse array in CSC format
        These arrays are the representations of the adjacency matrix
        that can accelerate the operations within this function.
        
    elt_indices, elt_indices_tr : 2D numpy array
        These arrays contain the indices of the non-zero values of A
        and the transpose of A.
        They are useful for speed optimization of the GT-Model dynamics.
    
    zeros, zeros1 : 2D numpy arrays
    zeros_vector, zeros_vector1, zeros_vector2, zeros_vector3 : 1D numpy arrays
        These arrays store the temporary results of cash flows, goods flows
        and the temporary results of next values of cash, goods and price.
    
    cash, goods, price : numpy arrays
        These arrays have the following shape (:, chunk_alpha_mu, 1)
        
    Returns
    ------
    cash, goods, price : numpy arrays
        These arrays contains the batch results of the simulations.
        They have the following shape (:, chunk_alpha_mu, chunk_epoch+1)
    
    """
    start_am = pair_am[0]
    end_am = pair_am[1]
#     print('start_am', start_am)
#     print('end_am', end_am)
    start_t = pair_t[0]
    end_t = pair_t[1]
    for am in range(0, end_am-start_am+1, 1):

        pair_alpha_mu = index_to_alpha_mu[start_am + am]
#         print(am)
#         print(pair_alpha_mu)
        
        alpha =  pair_alpha_mu[0] * 0.001
        mu = pair_alpha_mu[1] * 0.001
#         
#         print('alpha', alpha)
#         print('mu', mu)
#         print('====================')
        
        for t in range(0, end_t-start_t+1, 1):
            c = cash[:,am,t]
            g = goods[:,am,t]
            p = price[:,am,t]
            
#             print('t', t+1)
#             print(end_t-start_t)
#             print(c)
#             print(g)
#             print(p)
            
            length_c = c.shape[0]
            alpha_c = np.dot(alpha, c)
            mu_g = np.dot(mu, g)
            one_alpha_c = np.dot(1-alpha, c)
            one_mu_g = np.dot(1-mu, g)
            
            inv_p = 1 / p
            sum_inv_p = 1 / (csc_A * inv_p)
            c_compute_cash_flow(zeros, alpha_c * sum_inv_p, 
                inv_p, elt_indices)
            C_f = zeros
        
            #print('C_f=', C_f)
        
            C_f_tr = np.transpose(C_f)
        
            #print('C_f_tr=', C_f_tr)
        
            tl.c_rowsum(zeros_vector, C_f_tr, elt_indices_tr)
            sum_C_f_tr = zeros_vector
        
            #print('sum_C_f_tr=', sum_C_f_tr)
        
            d = mu_g * (1 / sum_C_f_tr)
            c_compute_goods_flow(zeros1, d, C_f_tr, elt_indices_tr)
            g_f = zeros1
        
            #print('G_f=', g_f)

            c_next = one_alpha_c + sum_C_f_tr
        
        
            g_f_tr = g_f.transpose()
            tl.c_rowsum(zeros_vector1, g_f_tr, elt_indices)
            sum_g_f_tr = zeros_vector1
        
            #print('sum_g_f_tr=', sum_g_f_tr)
        
            g_next = one_mu_g + sum_g_f_tr
        
            c_compute_price(zeros_vector2, C_f, g_f, elt_indices)
            # zeros_vector3 is important to avoid passing by reference
            # but make a real copy of zeros_vector2
            p_next = zeros_vector3 + zeros_vector2
            
            
#             print('c_next=', c_next)
#             print('g_next=', g_next)
#             print('p_next=', p_next)

            cash[:,am,t+1] = c_next
            goods[:,am,t+1] = g_next
            price[:,am,t+1] = p_next
    
    return cash, goods, price
