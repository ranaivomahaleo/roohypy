
cimport cython
cimport numpy as cnp
import gmpy2 as g2
import numpy as np

def cython_row_sum(
        cnp.ndarray[object, ndim=2] zeros,
        cnp.ndarray[object, ndim=2] input,
        cnp.ndarray[int, ndim=2] elt_indices
    ):
    cdef int m, i, j
    for m in range(elt_indices.shape[0]):
        i = elt_indices[m, 0]
        j = elt_indices[m, 1]
        zeros[i,0] = g2.mpfr('0')
    for m in range(elt_indices.shape[0]):
        i = elt_indices[m, 0]
        j = elt_indices[m, 1]
        zeros[i,0] = zeros[i,0] + input[i,j]
    return zeros 



def cython_adjmatrix_times_vector(
        int n,
        cnp.ndarray[object, ndim=2] vector,
        cnp.ndarray[int, ndim=2] adjmatrix_elt_indices
    ):
    cdef int m, i, j
    #zeros = g2.mpfr('0') * np.zeros((n,1))
    cdef cnp.ndarray arr = np.zeros([n, 1], dtype=object)
    for m in range(adjmatrix_elt_indices.shape[0]):
        i = adjmatrix_elt_indices[m, 0]
        j = adjmatrix_elt_indices[m, 1]
        arr[i,0] = arr[i,0] + vector[j,0]
    return arr



def cython_four_scalar_vector_multiplications(
        scalar1,
        cnp.ndarray[object, ndim=2] vector1,
        scalar2,
        cnp.ndarray[object, ndim=2] vector2,
        scalar3,
        cnp.ndarray[object, ndim=2] vector3,
        scalar4,
        cnp.ndarray[object, ndim=2] vector4
    ):
    cdef int m
    cdef cnp.ndarray arr1 = np.zeros([vector1.shape[0], 1], dtype=object)
    cdef cnp.ndarray arr2 = np.zeros([vector1.shape[0], 1], dtype=object)
    cdef cnp.ndarray arr3 = np.zeros([vector1.shape[0], 1], dtype=object)
    cdef cnp.ndarray arr4 = np.zeros([vector1.shape[0], 1], dtype=object)
    for m in range(vector1.shape[0]):
        arr1[m,0] = vector1[m,0] * scalar1
        arr2[m,0] = vector2[m,0] * scalar2
        arr3[m,0] = vector3[m,0] * scalar3
        arr4[m,0] = vector4[m,0] * scalar4  
    return arr1, arr2, arr3, arr4
    


def cython_scalar_vector_multiplication(
        scalar,
        cnp.ndarray[object, ndim=2] vector,
    ):
    cdef int m
    cdef cnp.ndarray arr = np.zeros([vector.shape[0], 1], dtype=object)
    for m in range(arr.shape[0]):
        arr[m,0] = vector[m,0] * scalar
    return arr



def cython_compute_cash_flow(
        cnp.ndarray[object, ndim=2] zeros,
        cnp.ndarray[object, ndim=2] D,
        cnp.ndarray[object, ndim=2] inv_p,
        cnp.ndarray[int, ndim=2] elt_indices
    ):
    cdef int m, i, j
    for m in range(elt_indices.shape[0]):
        i = elt_indices[m, 0]
        j = elt_indices[m, 1]
        zeros[i,j] = D[i,0] * inv_p[j,0]
    return zeros



def cython_compute_goods_flow(
        cnp.ndarray[object, ndim=2] zeros,
        cnp.ndarray[object, ndim=2] D,
        cnp.ndarray[object, ndim=2] C_f_tr,
        cnp.ndarray[int, ndim=2] elt_indices_tr
    ):
    cdef int m, i, j
    for m in range(elt_indices_tr.shape[0]):
        i = elt_indices_tr[m, 0]
        j = elt_indices_tr[m, 1]
        zeros[i,j] = D[i,0] * C_f_tr[i,j]
    return zeros


def cython_compute_price(
        cnp.ndarray[object, ndim=2] zeros,
        cnp.ndarray[object, ndim=2] cf,
        cnp.ndarray[object, ndim=2] gf,
        cnp.ndarray[int, ndim=2] elt_indices
    ):
    cdef int m, i, j
    for m in range(elt_indices.shape[0]):
        i = elt_indices[m, 0]
        j = elt_indices[m, 1]
        zeros[j,0] = cf[i,j] / gf[j,i]
    return zeros
