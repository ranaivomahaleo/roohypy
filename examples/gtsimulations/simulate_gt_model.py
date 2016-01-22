# !/usr/bin/python
# -*- coding=utf-8 -*-
# Python 2.7 script (bitshuffle uses python 2)
#    Copyright (C) 2015 by
#    Ranaivo Razakanirina <ranaivo.razakanirina@atety.com>
#    All rights reserved.
#    BSD license.

import roohypy.simulators as sim

# Network parameters and
# set manually some network attributes
# Here for example, we have an ER with 200 nodes and with p=0.2
network = {}
network['networkname'] = 'N200_p0.2_002'
network['networkfolder'] = './networks/' # With trailing slash

attributes = {}
attributes['p'] = 0.2
attributes['algorithm'] = 'ER'

# Launch a GT simulation corresponding to the above simulation
# network and attributes parameters.
sim.LaunchGTSimulation(network, attributes=attributes, simulation_index=0)

