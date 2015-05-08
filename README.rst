
RoohyPy
=======

RoohyPy is a simulator codes and libraries for discrete dynamical systems
built upon dynamical multiplex networks (network consisting of distinct
and multiple layers of interactions between the same set of agents).

RoohyPy implements the matricial representation of the dynamics.

Installation
------------

This simulator requires the following Python packages:

* NetworkX libraries

* numpy libraries for arrays manipulation

* scipy.weave for C codes that accelerate the treatment of 
  large sparse arrays

* bitshuffle and h5py packages for lossless compression algorithm 
  for hdf5 datasets that contain the results of the simulations.

To install the simulator, use the standard setup.py script to install 
a python package:

    python setup.py install

Simulator
---------

The simulator takes as input the parameters of the network and simulations
and stores the simulation results in a HDF5 compressed file.

GT-Model simulator
``````````````````
The GT-Model simulator is furnished within the package.
It models the interactions between homogeneous agents that exchange assets
against commodities.

::

import roohypy.simulators as sim

# Simulation parameters
simulation = {}
simulation['epochs'] = 100
simulation['alpha_mu_interval'] = 200
simulation['resultfolder'] = './results/' # With trailing slash
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
network['networkfolder'] = './networks/' # With trailing slash

attributes = {}
attributes['p'] = 0.2
attributes['algorithm'] = 'ER'

# Launch a GT simulation corresponding to the above simulation
# network and attributes parameters.
sim.LaunchGTSimulation(simulation, network, attributes=attributes)
