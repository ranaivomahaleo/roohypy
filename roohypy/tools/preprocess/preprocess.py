# !/usr/bine/python
# -*- coding = utf-8 -*-
# Python 2.7

import time
import networkx as nx
import roohypy.tools as tl
import roohypy as rp
import h5py as hdf
from bitshuffle import h5
import numpy as np


def getGTMultipleZFromMultipleDatasets(z_configurations):
    """
    This function returns the Z data of parameter basins
    z_configurations = np.array([
        [resultsroot, 'N200_p0.05_001_s20_is1000_i20000/', 'dataset_1.h5', 'cash', 0, 100, 0.01, 'z - 001', network_folder, True],
        [resultsroot, 'N200_p0.05_002_s20_is1000_i20000/', 'dataset_1.h5', 'cash', 0, 100, 0.01, 'z - 002', network_folder, True],
        [resultsroot, 'N200_p0.05_003_s20_is1000_i20000/', 'dataset_1.h5', 'cash', 0, 100, 0.01, 'z - 003', network_folder, True],
        [resultsroot, 'N200_p0.05_004_s20_is1000_i20000/', 'dataset_1.h5', 'cash', 0, 100, 0.01, 'z - 004', network_folder, True],
        [resultsroot, 'N200_p0.05_005_s20_is1000_i20000/', 'dataset_1.h5', 'cash', 0, 100, 0.01, 'z - 005', network_folder, True],
        [resultsroot, 'N200_p0.05_006_s20_is1000_i20000/', 'dataset_1.h5', 'cash', 0, 100, 0.01, 'z - 006', network_folder, True],
        [resultsroot, 'N200_p0.05_007_s20_is1000_i20000/', 'dataset_1.h5', 'cash', 0, 100, 0.01, 'z - 007', network_folder, True],
        [resultsroot, 'N200_p0.05_008_s20_is1000_i20000/', 'dataset_1.h5', 'cash', 0, 100, 0.01, 'z - 008', network_folder, True],
        [resultsroot, 'N200_p0.05_009_s20_is1000_i20000/', 'dataset_1.h5', 'cash', 0, 100, 0.01, 'z - 009', network_folder, True],
        [resultsroot, 'N200_p0.05_010_s20_is1000_i20000/', 'dataset_1.h5', 'cash', 0, 100, 0.01, 'z - 010', network_folder, True],
    ])
    """
    # Open resources and get data
    data = []
    for file in z_configurations:

        start_time = time.time()
        
        data_item = {}
    
        datasetfullpath = file[0] + file[1] + file[2]
        f = hdf.File(datasetfullpath, 'r')
        
        if int(file[9])==0:
            setid = False
        else:
            setid = True
        
        # If setid=True, the script get parameters basins of agent
        # defined by the columns #4 of the z_configurations.
        # Otherwise, we compare nodes with the highest
        # - left eigenvector centralities (importance of incoming neighbours)
        # - page rank (importance of incoming diluated in the outgoing neighbours)
        if setid==True:
            agent_id = int(file[4])
        else:
            networkname = f.attrs['networkname']
            networkfolder = file[8]
            nodefilepath = networkfolder + networkname + '/nodes.csv'
            edgefilepath = networkfolder + networkname + '/edges.csv'
            A, dgraph = tl.edgeNodeCsvToAdj(nodefilepath, edgefilepath)
            # rank = nx.eigenvector_centrality_numpy(dgraph)
            rank = nx.pagerank(dgraph, tol=1e-10)
            
            # Warning!!!: builtin max, not numpy max
            agent_id = max(rank.iterkeys(), key=(lambda k: rank[k]))
            
            txtinfo = ''
            txtinfo += 'Network ' + networkname + '\n'
            txtinfo += 'Agent with highest left eigenvector centrality ' + str(agent_id) + '\n'  
            print(txtinfo)      
        
        Z, X, Y, networkname = tl.getGTParameterBassinFromGTTransposedHdf5(f,
                                    last_epochs=int(file[5]),
                                    type=file[3],
                                    agent_id=agent_id,
                                    sensitivity=float(file[6]))
        
        data_item['Z'] = Z
        data_item['label'] = file[7]
        data_item['state'] = file[3] + str(agent_id)
        data.append(data_item)
    
        end_time = time.time()
        print('Iteration duration')
        print(end_time - start_time)
        
    return data


def getGTZmeanStdAll(datasetfullpath, type='price', last_epochs=100, sensitivity=0.01):
    """
    This function returns the all Z (from parameter basins) of all agents
    of a particular network.
    """
    f = hdf.File(datasetfullpath, 'r')
    
    integer_sensitivity = f.attrs['integer_sensitivity']
    
    agent_ids = np.array(f[type].dims[0][0])
    data = f[type][:,:,-last_epochs:]
    
#     agent_ids = np.array([0, 1])
#     data = np.array(
#         [[[1, 2, 3, 4],
#         [1, 2, 3, 4],
#         [1, 2, 3, 4]],
#         [[6, 6, 8, 9],
#         [6, 6, 8, 9],
#         [6, 6, 8, 9]]]
#     )
    
    # count unique value along the axis 2.
    start_time = time.time()
    Ks = np.apply_along_axis(rp.tools.countuniquevalue, 2, data, sensitivity=sensitivity)
    end_time = time.time()
    print('Unique value computation duration:')
    print(end_time - start_time)

    return Ks, agent_ids
    
