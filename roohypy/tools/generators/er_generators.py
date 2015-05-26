# !/usr/bin/python
# -*- coding=utf-8 -*-
# Python2 script

#    Copyright (C) 2015 by
#    Ranaivo Razakanirina <ranaivo.razakanirina@atety.com>
#    All rights reserved.
#    BSD license.

from __future__ import division

import networkx as nx
import roohypy.tools as tl
import roohypy.core as cr


def saveDigraphToFile(dgraph, folder, networkname, statinfo=''):
    """
    This function saves the directed graph digraph
    to nodes.csv and edges.csv files
    inside folder/networkname/ directory.
    
    These files can be used by Gephi to have 
    the graphical representation of the graph.
    
    Parameters
    ----------
    dgraph : networkx digraph
        The networkx digraph to save
    folder : string
        The folder name with trailing slash.
    networkname : string
        The network name
    
    Returns
    -------
    nodefilepath : string
        The full path of nodes.csv file.
    edgefilepath : string
        The full path of edges.csv file.
    statspath : string
        The full path of stats.txt file

    """
    folderPath = folder + networkname + '/'
    tl.checkCreateFolder(folderPath)
    
    nodefilepath = folderPath + 'nodes.csv'
    edgefilepath = folderPath + 'edges.csv'
    statspath = folderPath + 'stats.txt'
    
    with open(nodefilepath, 'w') as nodefile:
        nodefile.write('Nodes Id Label\n')
        for value in dgraph.nodes():
            nodefile.write(str(value) + ' ' + str(value) + ' ' + str(value) + '\n')
    nodefile.close()
    
    with open(edgefilepath, 'w') as edgefile:
        edgefile.write('Id Source Target\n')
        for key, value in enumerate(dgraph.edges()):
            edgefile.write(str(key) + ' ' + str(value[0]) + ' ' + str(value[1]) + '\n')
    edgefile.close()
    
    with open(statspath, 'w') as statsfile:
        statsfile.write(statinfo)
    statsfile.close()
    
    return nodefilepath, edgefilepath, statspath


def createERNetworks(folder, networkname, n=200, p=0.01):
    """
    This function is used to generate an ER network
    and store it in nodes.csv, edges.csv and stats.txt files
    in the directory folder/networkname.
    
    Parameters
    ----------
    folder : string
        The folder name with trailing slash.
    networkname : string
        The network name.
    n : int
        The number of nodes.
    p : float
        The connection probability between nodes.
    
    Returns
    -------
    nodefilepath : string
        The full path of nodes.csv file.
    edgefilepath : string
        The full path of edges.csv file.
    statspath : string
        The full path of stats.txt file
        
    """

    # Create ER graph and save edge, nodes and characteristics to files
    dgraph = nx.erdos_renyi_graph(n=n, p=p, directed=True)
    
    m = dgraph.number_of_edges()
    n_scc, n_cc = cr.getconnectedcomponents(dgraph)
    
    statinfo = 'Erdos random directed network\n'
    statinfo += 'n='+str(n)+', p='+str(p)+'\n'
    statinfo += 'm='+str(m)+'\n'
    statinfo += 'n_scc='+str(n_scc)+'\n'
    statinfo += 'n_cc='+str(n_cc)+'\n'
    
    nodefilepath, edgefilepath, statspath = tl.generators.saveDigraphToFile(
        dgraph, folder, networkname, statinfo=statinfo)
    
    return nodefilepath, edgefilepath, statspath
