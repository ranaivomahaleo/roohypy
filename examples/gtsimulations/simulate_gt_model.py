# !/usr/bin/python
# -*- coding=utf-8 -*-
# Python 2.7 script (bitshuffle uses python 2)
#    Copyright (C) 2015 by
#    Ranaivo Razakanirina <ranaivo.razakanirina@atety.com>
#    All rights reserved.
#    BSD license.

import roohypy.simulators as sim

# Simulation parameters
simulation = {}
simulation['epochs'] = 100
simulation['alpha_mu_interval'] = 200
simulation['resultfolder'] = './results/' # With trailing slash
simulation['rand_ic'] = False
simulation['c0'] = 300
simulation['g0'] = 40
simulation['p0'] = 10
simulation['alpha_mu_chunk_size'] = 16
simulation['epochs_chunk_size'] = 100
simulation['integer_sensitivity'] = 10000

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
sim.LaunchGTSimulation(simulation, network, attributes=attributes, simulation_index=0)

