# !/usr/bin/python
# -*- coding=utf-8 -*-
# Python 2.7 script (due to bitshuffle package)

#    Copyright (C) 2015 by
#    Ranaivo Razakanirina <ranaivo.razakanirina@atety.com>
#    All rights reserved.
#    BSD license.

import roohypy.tools.generators as gen

# Folder name with the trailing slash
folder = './networks/'

# Network name
networkname = 'N200_p0.5_001'

# Generate an Erdos Rényi network
gen.createERNetworks(folder, networkname, n=200, p=0.5)
