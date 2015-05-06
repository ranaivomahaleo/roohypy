# !/usr/bin/python
# -*- coding=utf-8 -*-
# Python 3

#    Copyright (C) 2015 by
#    Ranaivo Razakanirina <ranaivo.razakanirina@atety.com>
#    All rights reserved.
#    BSD license.

def GTModel(c, g, p, A, alpha, mu):
    """This function returns the t+1 values of a GT model
    """
    n_rows = A.shape[0]
    one = np.ones(n_rows)

    C_f = np.dot(
        np.diag(alpha * c),
        rows_sum_norm(
            np.dot(A, la.inv(np.diag(p)))
        )
    )

    G_f = np.dot(
        np.diag(mu*g),
        rows_sum_norm(np.transpose(C_f))
    )

    c_next = (one-alpha) * c + np.dot(np.transpose(C_f), one)
    g_next = (one-mu) * g + np.dot(np.transpose(G_f), one)
    p_next = np.dot(
        la.inv(np.diag(mu*g)),
        np.dot(np.transpose(C_f), one)
    )

    return c_next, g_next, p_next

