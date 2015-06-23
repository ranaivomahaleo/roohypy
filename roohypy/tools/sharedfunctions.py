# !/usr/bin/python
# -*- coding=utf-8 -*-
#    Copyright (C) 2015 by
#    Ranaivo Razakanirina <ranaivo.razakanirina@atety.com>
#    All rights reserved.
#    BSD license.

from __future__ import division
from collections import OrderedDict

import os.path
import networkx as nx
import numpy as np
import numpy.linalg as la
import time
import scipy.sparse as sparse
import scipy.weave as weave

import gmpy2 as g2

def checkFileOrFolder(fullpath):
    """
    """
    if os.path.exists(fullpath):
        return True
    else:
        return False


def checkCreateFolder(folderPath):
    """
    folderPath is with trailing slash
    """
    if not os.path.exists(folderPath):
        os.makedirs(folderPath)


def nparray2dataplot(x, y):
    """This function transforms two arrays having the same dimension x and y
    that are necessaries to be plotted to flat arrays able to be plotted.
    Eg:
    x = [0.05, 0.02, 0.03]
    y = [[2, 3, 7, 9], [-1, 6, 3], [7, 2]]
    The returned arrays are:
    datax = [0.05, 0.05, 0.05, 0.05, 0.02, 0.02, 0.02, 0.03, 0.03]
    datay = [2, 3, 7, 9, -1, 6, 3, 7, 2]
    """
    datax = sum(list(map(lambda x, y: [x] * y.shape[0], x, y)), [])
    datay = sum(y, [])
    return datax, datay
    

def array2dataplot(x, y):
    """This function transforms two arrays having the same dimension x and y
    that are necessaries to be plotted to flat arrays able to be plotted.
    Eg:
    x = [0.05, 0.02, 0.03]
    y = [[2, 3, 7, 9], [-1, 6, 3], [7, 2]]
    The returned arrays are:
    datax = [0.05, 0.05, 0.05, 0.05, 0.02, 0.02, 0.02, 0.03, 0.03]
    datay = [2, 3, 7, 9, -1, 6, 3, 7, 2]
    """
    datax = sum(list(map(lambda x, y: [x] * len(y), x, y)), [])
    datay = sum(y, [])
    return datax, datay
    
    
def dict2NpArray(initial_dict):
    """
    This function converts a dict with keys and values as arrays
    to a 3d np array. The values are returned sorted
    according to the key values.
    
    Input
    initial_dict = {
        (100,300) : np.array([10, 11, 12]),
        (400,100) : np.array([1, 10, 3])
    }
    
    Return np array in 3d (1 (epoch), agent_id, alpha_mu_indices)
    np.array([[
        [10, 11, 12],
        [1, 10, 3]
    ]])
    """
    ordered_dict = OrderedDict(sorted(initial_dict.items()))
    a = np.array(ordered_dict.values())
    return np.reshape(a, (1, a.shape[0], a.shape[1]))
    
    
def getIndexOf2DNpArray(array2D, x, y):
    """
    array2D: Numpy array
    array2D =
    [[ 10  10]
     [ 10  20]
     [ 10  30]
     ..., 
     [990 970]
     [990 980]
     [990 990]]
    This function returns 2 for x = 10, y = 30.
    and returns -1 if the combination of x and y does not exist.
    """
    for index, value in enumerate(array2D):
        if np.array_equal(value, np.array([x, y])):
            return index
    print('Error in tl.getIndexOf2DNpArray: Index does not exist')
    return -1


def listAllOrderedCombinations(list1, list2):
    """
    Return tuples of all possible combinations of list1 and list 2
    Let us consider
    list1 = [0, 1]
    list2 = [4, 5]
    This function returns
    [(0, 4), (0, 5), (1, 4), (1, 5)]
    """
    list_x_y = [(x, y) for x in list1 for y in list2]
    sorted_list_x_y = sorted(list_x_y)
    alpha_mu_to_index = {}
    index_to_alpha_mu = {}
    for i in sorted_list_x_y:
        alpha_mu_to_index[i] = sorted_list_x_y.index(i)
        index_to_alpha_mu[sorted_list_x_y.index(i)] = i
    alphas_mus_indices = list(map(lambda x: alpha_mu_to_index[x], sorted_list_x_y))
    return sorted_list_x_y, alphas_mus_indices, alpha_mu_to_index, index_to_alpha_mu


def getuniquesecondlevelkeylist(x):
    """This function returns the list of unique values of the second level key 
    of a dict.
    Let us consider the following dict.
    x = {0: {1: 50, 2: 20}, 1: {1: 300}}
    
    This function returns
    [1, 2]
    """
    y = list()
    for x_index in x:
        for y_index in x[x_index]:
            y.append(y_index)
    y = list(set(y)) 
    return y


def countuniquevalue(x, sensitivity=100000):
    """
    """
    k = list(map(lambda x: round(sensitivity*x), x))
    return len(set(k))
    
    
def chunkList(initialList, chunkSize):
    """
    This function chunks a list into sub lists 
    that have a length equals to chunkSize.
    
    Example:
    lst = [3, 4, 9, 7, 1, 1, 2, 3]
    print(chunkList(lst, 3)) 
    returns
    [[3, 4, 9], [7, 1, 1], [2, 3]]
    """
    finalList = []
    for i in range(0, len(initialList), chunkSize):
        finalList.append(initialList[i:i+chunkSize])
    return finalList


def getListOfNonNullIndex(A):
    """
    This function returns the 2d numpy array of the list of indices
    that corresponds to non-null value in A
    If
    A = array([[ 1.,  0.,  0.],
       [ 0.,  1.,  0.],
       [ 0.,  0.,  1.]])
    this function returns
    array([[0, 0],
       [1, 1],
       [2, 2]])
    """
    nzarray_cf = np.array(np.transpose(np.nonzero(A)))
    nzarray_gf = np.array(np.transpose(np.nonzero(np.transpose(A))))
    return nzarray_cf, nzarray_gf
    
    
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
    return A, dgraph
    
    
def getKINList(nodefilepath, edgefilepath):
    """
    Return the in_degree of each node of a directed graph.
    """
    dgraph = nx.DiGraph()
    
    nodes = np.genfromtxt(nodefilepath, skip_header=1, dtype=np.uint32)
    edges = np.genfromtxt(edgefilepath, skip_header=1, dtype=np.uint32)
    
    dgraph.add_nodes_from(nodes[:, 1])
    dgraph.add_edges_from(edges[:, 1:])
    
    return list(dgraph.in_degree(dgraph.nodes()).values())


def getHomogeneousInitialConditions(cini, gini, pini, n):
    """
    Return homogeneous initial conditions
    """
    #g2.get_context().precision = 400
    #g2.get_context().round=3
    
    ones = np.ones(n, dtype=object) 
    p_ic = pini * ones
    c_ic = cini * ones
    g_ic = gini * ones
    return c_ic, g_ic, p_ic
    
    
def getRandomUniformIC(
        c_tot=900,
        g_tot=120,
        alpha_mu_interval=20,
        c_min_lim=10,
        g_min_lim=10,
        integer_sensitivity=1000,
        alpha=100,
        mu=100,
        n=3
    ):
    """
    Return random uniform initial conditions.
    """    
    # Generate random number from Dirichlet pdf
    # that ensures sum up 1
    # Test 10 times if all c > c_min_lim / integer_sensitivity
    # AND all g > g_min_lim / integer_sensitivity
    for i in range(0, 10, 1):
        c_ic = np.dot(np.random.dirichlet(np.ones(n)*50, 1), c_tot)
        g_ic = np.dot(np.random.dirichlet(np.ones(n)*50, 1), g_tot)
        
        c_ic = c_ic.reshape((n,))
        g_ic = g_ic.reshape((n,))
        
        test_c = c_ic[c_ic <= c_min_lim / integer_sensitivity].size
        test_g = g_ic[g_ic <= g_min_lim / integer_sensitivity].size
        if int(test_c + test_g) == 0:
            break
        else:
            #If test fails after 10 times, use homogeneous initial conditions
            # for c and g (re-compute p_ic later)
            c_ic, g_ic, p_ic = getHomogeneousInitialConditions(c_tot/n, g_tot/n, 10, n)
            # print('c_ic and g_ic below a particular limit')

    # Compute p_min and p_max and generate U(p_min, p_max)
    alpha_min = alpha_mu_interval / 1000
    alpha_max = (1000 - alpha_mu_interval) / 1000
    mu_min = alpha_min
    mu_max = alpha_max
    p_min = (alpha_min * c_tot) / (mu_max * g_tot)
    p_max = (alpha_max * c_tot) / (mu_min * g_tot)
        
    p_ic = np.random.uniform(low=p_min, high=p_max, size=n)

#     print('c_ic', c_ic)
#     print('sum_c_ic', np.sum(c_ic))
#     print('g_ic', g_ic)
#     print('sum_g_ic', np.sum(g_ic))
#     print('p_ic', p_ic)
#     quit()
    
    return c_ic, g_ic, p_ic
    
    
def getICFromFile(icfile):
    """
    Get initial conditions from external file.
    The icfile should exist.
    """
    icdata = np.loadtxt(icfile)
    c_ic = icdata[:,0]
    g_ic = icdata[:,1]
    p_ic = icdata[:,2]
    
#     print('c_ic', c_ic)
#     print('sum_c_ic', np.sum(c_ic))
#     print('g_ic', g_ic)
#     print('sum_g_ic', np.sum(g_ic))
#     print('p_ic', p_ic)
#     quit()
    
    return c_ic, g_ic, p_ic


def getEpochsFromChunkIds(chunkidsarray, chunk_epoch):
    """
    This function gets an array of chunk ids
    and returns the complete array of corresponding epochs id.
    For instance with:
        chunkidsarray = [0, 6]
        chunk_epoch = [[0,99], ...[600, 699], ... [19900, 19999]]
        returned epochs array is: [0,...99,600,...699]
    """
    epochs_list = list()
    validchunkidsarray = list()
    for chunkid in chunkidsarray:
        # Consider only the existing chunk id.
        if chunkid < len(chunk_epoch):
            epochs_list = np.append(epochs_list, chunk_epoch[chunkid])
            validchunkidsarray.append(chunkid)
    return epochs_list, validchunkidsarray

def getResultFolderName(networkname='networkname',
                        step=10,
                        epochs=100,
                        integer_sensitivity=1):
    """
    """
    return (networkname + '_' + 's' + str(step)
        + '_is' + str(integer_sensitivity) + '_i' + str(epochs))


def c_rowsum(o, input, indices):
    code = \
    """
    int m, i, j;
    for (m=0; m<Nindices[0]; m++) {
        i = INDICES2(m,0);
        j = INDICES2(m,1);
        O1(i) = 0;
    }
    for (m=0; m<Nindices[0]; m++) {
        i = INDICES2(m,0);
        j = INDICES2(m,1);
        O1(i) = O1(i) + INPUT2(i,j);
    }
    """
    weave.inline(code, ['o', 'input', 'indices'])




