# !/usr/bin/python
# -*- coding=utf-8 -*-

#    Copyright (C) 2015 by
#    Ranaivo Razakanirina <ranaivo.razakanirina@atety.com>
#    All rights reserved.
#    BSD license.


from __future__ import division

import time
import networkx as nx
import numpy as np
import numpy.linalg as la
import roohypy.core as cr
import roohypy.tools as tl
import scipy.sparse as sparse
import scipy.weave as weave

import gmpy2 as g2

#import pyximport
# pyximport.install(
#     setup_args={
#         "include_dirs":[np.get_include()],
#     },
#     reload_support=True)
#pyximport.install()

import c_gtmodel

def getListOfAlphaMu(step_parameter_interval):
    """alpha and mu are the parameters of GT-Model.
    These parameters are between 0 and 1.
    We use an integer representation such that the real 1 becomes 1000,
    and the real 0.025 becomes 25.
    
    This function returns the ordered list of all possible combinations of 
    values of alpha and mu. Each combination is referenced using an integer
    index.
    
    Parameters
    ----------
    step_parameter_interval : int
        The interval size.
        step_parameter_interval = 25 means that we consider the following
        values of alpha and mu:
        0.025, 0.050, 0.075, 0.100, ... 0.975.
        The integer representation becomes
        25, 50, 75, 100, ..., 975
    
    Returns
    -------
     alphas_mus : numpy array
        Ordered list of all possible combinations of tuple (alpha, mu).
        Example: [(10, 10), (10, 20), (10, 30), ..., (990, 990)]
     
     alphas_mus_indices : numpy array
        Ordered list of the corresponding indices of each (alpha, mu) tuple.
     
     alpha_mu_to_index : dict
        Dict allowing the conversion of a tuple (alpha, mu) 
        to its corresponding index.
     
     index_to_alpha_mu : dict
        Dict allowing the conversion of an index to a tuple (alpha, mu).
    
    """
    mus = range(step_parameter_interval, 1000, step_parameter_interval)
    alphas = range(step_parameter_interval, 1000, step_parameter_interval)
    alphas_mus, alphas_mus_indices, alpha_mu_to_index, index_to_alpha_mu = tl.listAllOrderedCombinations(alphas, mus)
    return alphas_mus, alphas_mus_indices, alpha_mu_to_index, index_to_alpha_mu
    
    
def edgeNodeCsvToAdj(nodefilepath, edgefilepath):
    """Return the numpy adjacency matrix A
    corresponding to the nodes.csv and edge.csv files.
    
    Parameters
    ----------
    nodefilepath : string
        Full path of nodes.csv file
        
    edgefilepath : string
        Full path of edges.csv file
        
    Returns
    ------
    A : numpy array
        The adjacency matrix A
    
    """
    dgraph = nx.DiGraph()
    
    nodes = np.genfromtxt(nodefilepath, skip_header=1, dtype=np.uint32)
    edges = np.genfromtxt(edgefilepath, skip_header=1, dtype=np.uint32)
    
    dgraph.add_nodes_from(nodes[:, 1])
    dgraph.add_edges_from(edges[:, 1:])
    
    A = nx.to_numpy_matrix(dgraph, dtype=np.uint16)
    #A = np.array(A)
    
    return A
    
    
def getMainNetworkCharacteristics(A):
    """This function returns the characteristics of the directed network A
    
    Parameters
    ----------
    A : numpy 2D array
        The adjacency matrix of the considered directed network.
        
    Returns
    -------
    n : integer
        The order of the network A.
        
    csc_A : dict
        CSC sparse matrix notation of the adjacency matrix A.
        
    elt_indices, elt_indices_tr : numpy array
        These arrays contain the indices of the non-zero values of A
        and the transpose of A.
        
    attributes : dict
        Dict like attributes of the network A. The keys are:
            n : Network order.
            m : Network size.
            p_rand : Corresponding connection probability.
            directed : True (we always consider the case of directed graph
            in GT-Model).
            acyclic : True if A is directed acyclic graph and False if A 
            is a directed cyclic graph.
            n_scc : Number of strongly connected components.
            n_cc : Number of connected components.
            title, networkname, p, m0, m_links, beta, k, algorithms : Keys
            that will be used when generating random graphs.

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
    attributes['networkname'] = ''
    attributes['p'] = 0
    attributes['m0'] = 0
    attributes['m_links'] = 0
    attributes['beta'] = 0
    attributes['k'] = 0
    attributes['algorithm'] = ''

    return n, csc_A, elt_indices, elt_indices_tr, attributes


def getNullArraysAndVectors(n):
    """This function returns two arrays and four vectors
    that are used as temporary storage for the optimized GT-Model dynamics.
    """
    
    #g2.get_context().precision = 100
    
    zeros = g2.mpfr('0') * np.zeros((n, n))

    zeros1 = g2.mpfr('0') * np.zeros((n, n))
    
    zeros_vector = g2.mpfr('0') * np.zeros((n,1))
    
    zeros_vector1 = g2.mpfr('0') * np.zeros((n,1))
    
    zeros_vector2 = g2.mpfr('0') * np.zeros((n,1))
    
    zeros_vector3 = g2.mpfr('0') * np.zeros((n,1))

    return zeros, zeros1, zeros_vector, zeros_vector1, zeros_vector2, zeros_vector3
    

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
        cash, goods, price,
        n):
    """This function is the optimized version (v6) of the GT-Model dynamics.
    
    The inputs are the cash, goods and price arrays with shape
    (:, chunk_alpha_mu, 1)
    This function returns in batch the time evolution
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
    
    #g2.get_context().precision = 100
    #g2.get_context().round=2
    
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
        
        alpha =  g2.mpfr(pair_alpha_mu[0]) * g2.mpfr('0.001')
        mu = g2.mpfr(pair_alpha_mu[1]) * g2.mpfr('0.001')
        
#         print('alpha', alpha)
#         print('mu', mu)
#         print('====================')
#         quit()
        
        for t in range(0, end_t-start_t+1, 1):

            c = (cash[:,am,t]).reshape((n,1))
            g = (goods[:,am,t]).reshape((n,1))
            p = (price[:,am,t]).reshape((n,1))
                        
            alpha_c, \
            mu_g, \
            one_alpha_c, \
            one_mu_g = c_gtmodel.cython_four_scalar_vector_multiplications(
                            alpha, c,
                            mu, g,
                            g2.mpfr('1')-alpha, c,
                            g2.mpfr('1')-mu, g
            )

            inv_p = g2.mpfr('1') / p
            temp = c_gtmodel.cython_adjmatrix_times_vector(
                        n, inv_p, elt_indices)
            sum_inv_p =  g2.mpfr('1') / temp

            D = np.multiply(alpha_c, sum_inv_p)
            
            # elt_indices is a np array with int elements
            C_f = c_gtmodel.cython_compute_cash_flow(zeros, D, inv_p, elt_indices)
            C_f_tr = np.transpose(C_f)

            sum_C_f_tr = c_gtmodel.cython_row_sum(zeros_vector, C_f_tr, elt_indices_tr)
            c_next = one_alpha_c + sum_C_f_tr

            D = np.multiply(mu_g, g2.mpfr('1') / sum_C_f_tr)
            g_f = c_gtmodel.cython_compute_goods_flow(zeros1, D, C_f_tr, elt_indices_tr)
            g_f_tr = g_f.transpose()
            sum_g_f_tr = c_gtmodel.cython_row_sum(zeros_vector1, g_f_tr, elt_indices)
            g_next = one_mu_g + sum_g_f_tr
            
            p_next = c_gtmodel.cython_compute_price(zeros_vector2, C_f, g_f, elt_indices)
            
#             print('c_next=', c_next)
#             print('g_next=', g_next)
#             print('p_next=', p_next)
#             quit()
            
            cash[:,am,t+1] = c_next.reshape(n)
            goods[:,am,t+1] = g_next.reshape(n)
            price[:,am,t+1] = p_next.reshape(n)
    
    return cash, goods, price
