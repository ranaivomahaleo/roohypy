# !/usr/bin/python
# -*- coding=utf-8 -*-
# Python 2.7 script (bitshuffle uses python 2)
#    Copyright (C) 2015 by
#    Ranaivo Razakanirina <ranaivo.razakanirina@atety.com>
#    All rights reserved.
#    BSD license.

import h5py as hdf
from bitshuffle import h5 # bishuffle is mandatory for data decompression
import roohypy.tools as tl

# Path of the dataset
datasetfullpath = './results/N200_p0.2_002_s200_is10000_i100/dataset_0.h5'

# Read the hdf5 dataset
f = hdf.File(datasetfullpath, 'r')

# Get the GT simulations results 
# corresponding to alpha = 600 (0.6) and mu = 400 (0.4)
alpha = 600
mu = 400

# - The first line gets all possible combinations of alpha and mu
# stored in the dataset.
# - The second line transforms the combination of alpha and mu to
# its corresponding integer index.
# - The third line gets the assets ('cash' key) of traders 0 to 4
# from t=0 to t=9
alphas_mus = f['cash'].dims[1][0]
index_alpha_mu = tl.getIndexOf2DNpArray(alphas_mus, alpha, mu)
assets = f['cash'][0:5, index_alpha_mu, 0:10]

print(assets)
