

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

::
    python setup.py install

Due to the integration of pyx files, install the package in develop mode
::
    python setup.py develop
(see https://github.com/ranaivomahaleo/roohypy/issues/1 for more explanation about this hot fix).

Simulator
---------

The simulator takes as input the parameters of the network and simulations
and stores the simulation results in a HDF5 compressed file.

GT-Model (Give and Take Model) simulator
``````````````````
The GT-Model simulator is furnished within the package.
It models the interactions between homogeneous traders that exchange assets
against commodities.

1. R.M. Razakanirina, B Chopard, "Dynamics of Artificial Markets on Irregular Topologies", Proceedings of the European Conference on Complex Systems 2012, Springer, 1019-1031, 2013 [DOI1_]

.. _DOI1: http://dx.doi.org/10.1007/978-3-319-00395-5_123

2. R.M. Razakanirina, B. Chopard, "Using Cellular Automata on a Graph to Model the Exchanges of Cash and Goods", ACRI 2010, vol. 6350/2010, Ascoli Piceno, Springer, pp. 163-172, 21/09/2010. [DOI2_]

.. _DOI2: http://dx.doi.org/10.1007/978-3-642-15979-4_18

Sample Python code
:::::::::::
The Python code below simulates a GT-Model with the following 
simulation configurations (default configuration):

* :code:`epochs=100`: 100 iterations
* :code:`alpha_mu_interval=200`: Each parameters :code:`alpha` 
  and :code:`mu` varies
  from [200, 1000[ with the step interval 200 (200, 400, 600, 800).
  1000 is excluded.
* :code:`resultfolder='./results/'`: Defines where the results will be stored: './results/'
* :code:`c0=300`, :code:`g0=40` and :code:`p0=10`: Homogeneous
  initial conditions of each trader.
* :code:`alpha_mu_chunk_size=16`: dataset chunk size in alpha and mu axis
  (see figure below) 
* :code:`epochs_chunk_size=100`: dataset chunk size in epoch axis 
  (see figure below) defines the chunk

  
The network parameters are:

* :code:`networkfolder` which is the root folder where the network is stored
  in file system.
* :code:`networkname` which is the name of the simulated network.
  This name should correspond exactly to the folder inside
  the :code:`networkfolder`.
  This folder contains two files: nodes.csv and edges.csv files.
  
The structure of nodes.csv file is as follows:
  
::
  
    Nodes Id Label
    0 0 0
    1 1 1
    2 2 2
  
The structure of edges.csv file is as follows:
  
::
  
    Id Source Target
    0 0 2
    1 0 10
    2 0 15
    3 0 17
    4 0 24

The GT-Model simulation code is:

::

    # Code available at
    # roohypy/examples/gtsimulations/simulate_gt_model.py
    
    import roohypy.simulators as sim

    # Network parameters and
    # set manually some network attributes
    # Here for example, we have an ER with 200 nodes and with p=0.2
    network = {}
    network['networkname'] = 'N200_p0.2_002'
    network['networkfolder'] = './networks/' # With trailing slash

    attributes = {}
    attributes['p'] = 0.2
    attributes['algorithm'] = 'ER'

    # Launch a GT simulation corresponding to the default simulation configurations, 
    # network and attributes parameters.
    sim.LaunchGTSimulation(network, attributes=attributes, simulation_index=0)  


Structure of the resulting dataset of GT-Model
:::::::::::::::::::::::::::::::::::

The filename of the resulting dataset is :code:`dataset_'+simulation_index+'.h5`.
This file is stored inside the folder 

::

    resultfolder + networkname + _s'alpha_mu_interval' + _is'integer_sensitivity' + _i'epochs'

The resulting dataset consists of three subsets.
The first one for assets with :code:`cash` key,
the second one for commodities with :code:`goods` key
and the last one for prices with :code:`price` key.

Each subset has the shape :code:`(n_agents, alpha_mu, epochs)` as
depicted in the following figure:

.. image:: docs/images/gtdataset.png

Get data from the resulting dataset of GT-Model
:::::::::::::::::::::::::::::::::::

The following Python code explains how to extract data from 
the resulting dataset.
Notice that bitshuffle should imported with :code:`from bitshuffle import h5`
even not used within the code.

::

    # Code available at: 
    # roohypy/examples/gtsimulations/get_gt_data_from_dataset.py
    
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


The above code returns the following results
(the hdf5 dataset is available at
:code:`roohypy/examples/gtsimulations/results/N200_p0.2_002_s200_is10000_i100/dataset_0.h5`):

::

    [[ 300.          312.96763101  304.91503735  313.50568624  308.4347281
       314.17414831  310.98049562  314.85295998  312.83239204  315.48649578]
     [ 300.          315.30861274  308.27477492  318.15867155  313.75594732
       320.91645355  317.58452923  323.26142142  320.38308709  325.14130738]
     [ 300.          301.43608116  296.52577512  302.15469169  294.26647784
       301.89450917  292.70615126  301.18550113  291.58537403  300.30110894]
     [ 300.          305.89045415  303.16941947  306.73070302  305.09092535
       307.77706535  306.3621142   308.78892811  307.28123795  309.68173078]
     [ 300.          286.12310366  293.90219793  283.52655282  289.61854587
       281.29112037  286.57039924  279.4292609   284.36747032  277.91597793]]

