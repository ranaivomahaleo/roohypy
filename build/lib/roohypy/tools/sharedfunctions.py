# !/usr/bin/python
# -*- coding=utf-8 -*-
# Python 3

#    Copyright (C) 2015 by
#    Ranaivo Razakanirina <ranaivo.razakanirina@atety.com>
#    All rights reserved.
#    BSD license.

import numpy as np
import numpy.linalg as la

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


def rows_sum_norm(A):
    """This function returns the normalized version of row-sum
    of a square matrix.

    Parameters
    ----
    A: numpy array
      An input square matrix
    Returns
    ----
    Return diag^(-1)(A1).A
    """
    n_rows = A.shape[0]
    one = np.ones(n_rows)
    return np.dot(la.inv(np.diag(np.dot(A, one))), A)
