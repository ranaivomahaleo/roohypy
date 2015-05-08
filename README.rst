
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

To install the simulator, use the standard setup.py script to install a python package:

    python setup.py install


