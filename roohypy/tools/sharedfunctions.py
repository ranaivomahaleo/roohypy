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
    return A


def getHomogeneousInitialConditions(cini, gini, pini, n):
    """
    """
    p_ic = pini * np.ones(n)
    c_ic = cini * np.ones(n)
    g_ic = gini * np.ones(n)
    return c_ic, g_ic, p_ic


def getResultFolderName(networkname='networkname', step=10, epochs=100):
    """
    """
    return networkname + '_' + 's' + str(step) + '_i' + str(epochs)


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


# def rows_sum_norm(A):
#     """This function returns the normalized version of row-sum
#     of a square matrix.
# 
#     Parameters
#     ----
#     A: numpy array
#       An input square matrix
#     Returns
#     ----
#     Return diag^(-1)(A1).A
#     """
#     n_rows = A.shape[0]
#     one = np.ones(n_rows)
# #     a1 = reduce(np.dot, [A, one])
# #     diag_inv_a1 = la.inv(np.diag(a1))
# #     return reduce(np.dot, [diag_inv_a1, A])
#     
#     return np.dot(la.inv(np.diag(np.dot(A, one))), A)
    
    
# def c_rowsum(o, input, indices):
#     code = \
#     """
#     int m, i, j;
#     for (m=0; m<Nindices[0]; m++) {
#         i = INDICES2(m,0);
#         j = INDICES2(m,1);
#         O1(i) = 0;
#     }
#     for (m=0; m<Nindices[0]; m++) {
#         i = INDICES2(m,0);
#         j = INDICES2(m,1);
#         O1(i) = O1(i) + INPUT2(i,j);
#     }
#     """
#     weave.inline(code, ['o', 'input', 'indices'])
# 
# 
# def vector_to_square_matrix(v):
#     """
#     Example:
#     v = np.array([ 0.5  0.5  0.5  1. ])
#     
#     Return
#     [[ 0.5  0.5  0.5  0.5]
#      [ 0.5  0.5  0.5  0.5]
#      [ 0.5  0.5  0.5  0.5]
#      [ 1.   1.   1.   1. ]]
#     """
#     n_rows = v.shape[0]
#     return np.transpose(v * np.ones((n_rows, 1)))
# 
# 
# def vector_to_square_row(v, csc_res_one):
#     """
#     Example:
#     v = np.array([ 0.5  0.5  0.5  1. ])
#     csc_res_one has shape (length_v, 1)
#     
#     Return
#     [[ 0.5  0.5  0.5  1.]
#      [ 0.5  0.5  0.5  1.
#      [ 0.5  0.5  0.5  1.]
#      [ 0.5  0.5  0.5  1. ]]
#     """
#     length_v = v.shape[0]
#     result = csc_res_one * v.reshape(1, length_v)
#     return  result
# 
# 
# def rows_sum_norm_opt(A):
#     """
#     """
#     n_rows = A.shape[0]
#     div_sum_A = 1 / np.sum(A, axis=1)
#     row_sum_norm = np.multiply(A, vector_to_square_matrix(div_sum_A))
#     return row_sum_norm
# 
# 
# def optimized_rows_sum_norm_jit(A, nzarray):
# 
#     n_rows = A.shape[0]
#     nz_rows = nzarray.shape[0]
#     row_sum = np.zeros(n_rows)
#     row_sum_norm = np.zeros((n_rows, n_rows))
#     
#     optimized_rows_sum_norm(row_sum_norm, row_sum, n_rows, A, nzarray, nz_rows)
# 
#     return row_sum_norm
# 
# 
# def optimized_rows_sum_norm(row_sum_norm, row_sum, n_rows, A, nzarray, nz_rows):
#     """This function returns the normalized version of row-sum
#     of a square matrix (optimized version for JIT compiler).
# 
#     Parameters
#     ----
#     A: numpy array
#       An input square matrix
#     Returns
#     ----
#     Return diag^(-1)(A1).A
#     """    
#     for i in range(nz_rows):
#         row_sum[nzarray[i,0]] = row_sum[nzarray[i,0]] \
#             + A[nzarray[i,0], nzarray[i,1]]
#     
#     for i in range(nz_rows):
#         row_sum_norm[nzarray[i,0], nzarray[i,1]] = \
#             A[nzarray[i,0], nzarray[i,1]] / row_sum[nzarray[i,0]]


# def metafilename(initial_conditions_c=(0.8, 0.1, 0.1),
#                  initial_conditions_g=(0.8, 0.1, 0.1),
#                  step=0.005):
#     """
#     """
#     complete_meta = ('N1_meta_c123_'
#                           + str(round(initial_conditions_c[0], 5)) + '_' 
#                           + str(round(initial_conditions_c[1], 5)) + '_'
#                           + str(round(initial_conditions_c[2], 5)) + '_'
#                           + 'g123_'
#                           + str(round(initial_conditions_g[0], 5)) + '_' 
#                           + str(round(initial_conditions_g[1], 5)) + '_'
#                           + str(round(initial_conditions_g[2], 5))
#                           + '_s'
#                           + str(step))
#     complete_meta_folder  = complete_meta + '/'
#     complete_meta_filename = complete_meta + '.json'
#     
#     return complete_meta, complete_meta_folder, complete_meta_filename
# 
# 
# def metafilenamecomplete(networkname='N1',
#                          initial_conditions_c=(0.8, 0.1, 0.1),
#                          initial_conditions_g=(0.8, 0.1, 0.1),
#                          step=0.005,
#                          iteration=100):
#     """
#     """
#     c_indices = 'c' + "".join(map(lambda x:str(x), list(range(0,len(initial_conditions_c),1))))
#     g_indices = 'g' + "".join(map(lambda x:str(x), list(range(0,len(initial_conditions_g),1))))
#     c_ini = "_".join(map(lambda x: str(round(x,5)),initial_conditions_c))
#     g_ini = "_".join(map(lambda x: str(round(x,5)),initial_conditions_g))
#     
#     complete_meta = (networkname
#                     + '_s'
#                     + str(step)
#                     + '_i'
#                     + str(iteration))
#     complete_meta_folder  = complete_meta + '/'
#     complete_meta_filename = 'meta.txt'
#     
#     return complete_meta, complete_meta_folder, complete_meta_filename
    

