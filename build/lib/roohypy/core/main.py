# !/usr/bin/python
# -*- coding=utf-8 -*-
# Python3 script


#    Copyright (C) 2015 by
#    Ranaivo Razakanirina <ranaivo.razakanirina@atety.com>
#    All rights reserved.
#    BSD license.


# This library denoted mcag contains some functions that help on stats, graph,...

import statsmodels.distributions.empirical_distribution as sm
import numpy as np
import networkx as nx
import json as json
import sys
import os
import re
import itertools
import random


# Declare and init global variables
global mainrootfolder
global gtfolder
global lgmfolder
global outputfolder
global launcherfullpath
global gtexec

mainrootfolder    = '/Volumes/Data/phd/MCAGFramework/build/macosx/lendredeemcutmodel/'
gtfolder          = mainrootfolder+'gtmodelinput/'
lgmfolder         = mainrootfolder+'lgmmodelinput/'
outputfolder      = mainrootfolder+'outputfiles/'
launcherfullpath  = mainrootfolder+'launcher.py'
gtexec            = mainrootfolder+'gtmodel'


def updatemainrootfolder(newmainrootfolder):
  """This function updates the main root folder
  """
  # These global variable should be respecified
  # Otherwise, this update process does not work in python
  global mainrootfolder
  global gtfolder
  global lgmfolder
  global outputfolder
  global launcherfullpath
  global gtexec

  mainrootfolder    = newmainrootfolder
  gtfolder          = mainrootfolder+'gtmodelinput/'
  lgmfolder         = mainrootfolder+'lgmmodelinput/'
  outputfolder      = mainrootfolder+'outputfiles/'
  launcherfullpath  = mainrootfolder+'launcher.py'
  gtexec            = mainrootfolder+'gtmodel'



def lgm_outputs_fullpath(modelname, tau, alpha):
  """Create node and edges full paths and also the name of the resultfile
  considering tau and alpha and the different subfolders. 
  """
  
  resultname          = 'lgm_'+modelname+'_rate0_itau0_tau'+str(tau)+'_alpha'+str(alpha)
  model_output_folder = outputfolder+'lgm_'+modelname+'/'+resultname+'/'
  
  nodes_fullpath      = model_output_folder+'evolution_lgm_'+modelname+'.txt'
  edges_fullpath      = model_output_folder+'edge_evolution_lgm_'+modelname+'.txt'
  
  return model_output_folder, resultname, nodes_fullpath, edges_fullpath
  
  
  

  
def lrcim_outputs_fullpath(modelname, r, itau, alpha, tau=0):
  """Creates the nodes state full path and returns the folder where nodes states
  is stored
  """
  resultname          = 'lrcim_'+modelname+'_rate'+str(r)+'_itau'+str(itau)+'_tau'+str(tau)+'_alpha'+str(alpha)
  model_output_folder = outputfolder+'lrcim_'+modelname+'/'+resultname+'/'
  
  nodes_fullpath      = model_output_folder+'evolution_lrcim_'+modelname+'.txt'  
  return model_output_folder, nodes_fullpath
  




def gt_outputs_fullpath(modelname):
  
  model_output_folder = outputfolder+'gt_'+modelname+'/'
  
  nodes_fullpath      = model_output_folder+'evolution_gt_'+modelname+'.txt'
  edges_fullpath      = model_output_folder+'edge_evolution_gt_'+modelname+'.txt'

  return model_output_folder, nodes_fullpath, edges_fullpath





####
# Compute the empirical survival function
# The empirical data should by >=0
# and return the coordinates x and y of the result
# This function needs statsmodels (sm) and numpy (numpy)
####
def esf(data, side='left'):
  
  ecdf = sm.ECDF(data, side=side)
  # add 0 to the list of data
  # data.append(0)
  x = np.unique(data)
  y = 1-ecdf(x)
  return (x,y)
  
  


  
####
# Compute the probability mass function (normalized)
# of empirical distribution of integers (e.g. degree distribution)
####
def pmf(data):
  
  p = {}
  for i in data:
    if i in p.keys():
      p[i] = p[i] + 1
    else:
      p[i] = 1
    
  x = list(p.keys())
  y = list(p.values())
  y = [i/sum(y) for i in y]
  
  # Sort x and sync y
  x,y = zip(*sorted(zip(x, y)))
  return (list(x),list(y))
  
  


  
####
# Compute the pdf of real data
####
def pdf(realdata):
  
  hist, bins = np.histogram(realdata, bins=150, density=True)
  dis_return_max = np.amax(hist)
  width = 0.7 * (bins[1] - bins[0])
  center = (bins[:-1] + bins[1:]) / 2
  return (center,hist)
  
  
  
  
  

####
# Compute the distribution of scc of a directed graph
# Use pmf
####
def sccpmf(dgraph):
  
  number_of_scc = nx.number_strongly_connected_components(dgraph)
  sccs = nx.strongly_connected_components(dgraph)
  scc_sizes = [len(i) for i in sccs]
  x,y = pmf(scc_sizes)
  return (x,y)
  
  
  

def ccpmf(dgraph):
  """Compute the distribution of connected components of a directed graph.
  This function uses self.pmf
  
  Parameters
  ----
  dgraph : networkx digraph
    A directed graph
    
  Returns
  ----
  x : int
    x-axis of the distribution of component sizes
  y : int
    y-axis or frequency of the distribution of component sizes
  ncc : int
    Number of connected components
  nscc : int
    Number of strongly connected components
  
  """
  
  nscc = nx.number_strongly_connected_components(dgraph)
  graph = dgraph.to_undirected()
  ncc = nx.number_connected_components(graph)
  components = list(nx.connected_components(graph))
  components_sizes = [len(i) for i in components]
  x,y = pmf(components_sizes)
  return (x,y,ncc,nscc)
  
  
  
  
  
  
def clusterstats(dgraph):
  """To analyse the distribution of cluster of a directed graph, this script returns
  the relative size of the largest component and the average size of the isolated
  cluster. Here, size is the number of nodes.
  
  Parameters
  ----
  dgraph : Networkx directed graph
    The directed graph to analyse
  
  Returns
  ----
  giant_size : double
    The relative size of the largest component
    
  average_size : double
    The average size of the isolated cluster (components appart the largest component)
  
  """

  ugraph  = dgraph.to_undirected()
  
  gcc = sorted(nx.connected_components(ugraph), key=len, reverse=True)
  giant_size = len(gcc[0])/ugraph.number_of_nodes()
  
  # Giant directed graph
  giant_dgraph = dgraph.subgraph(gcc[0])
  
  # avg_s_path = nx.average_shortest_path_length(giant_dgraph)  
  
  # Remove the nodes that are part of giant component for the average statistic
  ugraph.remove_nodes_from(gcc[0])
  if not ugraph.number_of_nodes()==0:
    subcomponents = nx.connected_components(ugraph)
    average_size = np.mean([len(subcomponent) for subcomponent in subcomponents])
  else:
    average_size = 0
    
  #return giant_size, average_size, avg_s_path
  return giant_size, average_size
  
  
  
  
  
  
####
# Compute the distribution of in and out degree of directed graph
# Use pmf
####
def kpmf(dgraph):
  in_degree = list(dgraph.in_degree().values())
  out_degree = list(dgraph.out_degree().values())
  
  # Remove the distribution with in_degree or out_degree=0
  in_degree = [value for value in in_degree if value!=0]
  out_degree = [value for value in out_degree if value!=0]
  
  xin, yin = pmf(in_degree)
  xout, yout = pmf(out_degree)
  return (xin, yin, xout, yout)
  
  
  
  



def kesf(dgraph):
  """Compute the complementary cumulative distribution function of directed graph
  
  Parameters
  ----
  dgraph : Networkx directed graph
    The directed graph to compute
  """
  in_degree = list(dgraph.in_degree().values())
  out_degree = list(dgraph.out_degree().values())
  
  # Remove the distribution with in_degree or out_degree=0
  in_degree = [value for value in in_degree if value!=0]
  out_degree = [value for value in out_degree if value!=0]

  xin, yin = esf(in_degree)
  xout, yout = esf(out_degree)

  return (xin, yin, xout, yout)







####
# Transform an undirected graph to directed graph
# by passing through all the edge and respecting the existing 
####
def todirected(graph):
  dgraph = nx.DiGraph()
  dgraph.add_nodes_from(graph.nodes())
  dgraph.add_edges_from(graph.edges())
  return dgraph
  
  
  
  
  
  
####
# Import the list of nodes at each time iteration t
# Output: nodes_t[time] -> [1, 2, 3] or []
####
def getlistofnodes(nodes_fullpath):

  with open(nodes_fullpath, 'r') as nodesfile:
    data = json.load(nodesfile)
  
    nodes_t = {}
    for i in data:
      nodes = []
      for j in data[i]:
        nodes.append(int(j))
      nodes_t[int(i)] = nodes

  nodesfile.close()
  return nodes_t
  
  
  
  
####
# Import the state of each node at each time iteration t
# Output: nodes_t[time] -> {'cash': [20000,14000,50000], 'production': [46,3,10], ...}
####
def getstatetofnodes(nodes_fullpath):

  with open(nodes_fullpath, 'r') as nodesfile:    
    data = json.load(nodesfile)
  
    nodes_t = {}
    # Time level
    for i in data:
      nodes_t[int(i)] = {}
      # node id level
      for j in data[i]:      
        # key property level
        for k in data[i][j]:
          # k is the list of keys ("cash", "goods", "price", ...)
          if not k in nodes_t[int(i)].keys():
            tmp = []
          else:
            tmp = nodes_t[int(i)][k]
          tmp.append(data[i][j][k])
          nodes_t[int(i)][k] = tmp
          

  nodesfile.close()
  return nodes_t
  
  
  
  
  
  
  
def getstatetofnodesdict(nodes_fullpath):
  """Get the state of each node as with self.getstatetofnodes but returning
  a dictionary of corresponding key, values instead of a simple list
  
  Parameters
  ----
  nodes_fullpath : string
    Full path of the evolution node file (lgm)
    
  Returns
  ----
  Node dictionary with the following structure:
    nodes_t[time][key_state][node_id]
    
    nodes_t[200]["cash"] -> {0:20000, 1:10000, ...} = cash of all agents at time t=200
  
  """
  
  with open(nodes_fullpath, 'r') as nodesfile:
    data = json.load(nodesfile)
    
    nodes_t = {}
    
    # Time level
    for i in data:
      # node id level
      for j in data[i]:
        # key property level
        for k in data[i][j]:
        
          if not int(i) in nodes_t.keys():
            nodes_t[int(i)] = {}
          if not k in nodes_t[int(i)].keys():
            nodes_t[int(i)][k] = {}
          
          nodes_t[int(i)][k][int(j)] = data[i][j][k]

    nodesfile.close()
  return nodes_t
  
  
  
  
  
  
  
  
def getreversestatetofnodes(nodes_fullpath):
  """Return the state of each nodes versus time
  
  Parameters
  ----
  nodes_fullpath : string
    Full path of the evolution node file (lgm)
    
  Returns
  ----
  Node dictionary with the following structure:
    nodes_t[node_id][key_state][time]
    
    nodes_t[40]["cash"] -> {"0":40000,"1":10000,"2":10000,...} cash of node 40 versus time
    
  """
  
  with open(nodes_fullpath, 'r') as nodesfile:
    data = json.load(nodesfile)
    
    nodes_t = {}
    
    # Time level
    for i in data:
      # node id level
      for j in data[i]:
        # key property level
        for k in data[i][j]:
          
          if not int(j) in nodes_t.keys():
            nodes_t[int(j)] = {}
          
          if not k in nodes_t[int(j)].keys():
            nodes_t[int(j)][k] = {}
          
          nodes_t[int(j)][k][int(i)] = data[i][j][k]
  nodesfile.close()
  
  return nodes_t




####
# Import the list of edges at each time iteration t
# Output: edges_t[time][layer] -> [(1,2), (3,4)] or []
####
def getlistofedges(edges_fullpath):
  
  with open(edges_fullpath, 'r') as edgesfile:
    data = json.load(edgesfile)
  
    edges_t = {}
    # Time level
    for i in data:
      # Layer level
      edges_l = {}
      for j in data[i]:
        # Edge level
        edges = []
        for k in data[i][j]:
          start = int(data[i][j][k]["start"])
          end = int(data[i][j][k]["end"])
          edges.append((start,end))
        
        edges_l[int(j)] = edges
      edges_t[int(i)] = edges_l
  edgesfile.close()
  return edges_t





#### 2014-12-12
#### TODO: Add some stats inside textual info
#### TODO: Implement the choice to remove sink and source nodes.
def create_gt_erdos_graphs(n,
                    p,
                    removesinknodes=False,
                    removesourcenodes=False,
                    lmb_i=0.6,
                    mu_i=0.4,
                    phi_i=0,        # consumption_factor
                    pi_i=0,         # production_factor
                    node_type_i=0,  # node_type (unique type of nodes)
                    c_ini=50000,
                    g_ini=40,
                    p_ini=120):
  """Create GT-Model dgraph with default parameters.

    Parameters
    ----
    n : int
      Number of nodes
      
    p : float
      Connection probability between distinct pair of nodes.
      
    removesinknodes : boolean
      True if sink nodes are removed from the initial topology.
      False if the sink nodes are not removed from the initial topology.
      
    removesinknodes : boolean
      True if source nodes are removed from the initial topology
      False if the source nodes are not removed from the initial topology
  
    Returns
    ----
    dgraph : A directed networkx graph without multiedges and self loops
      Each node of this dgraph has the following attributes
        lmb, mu, cash, goods, price, consumption_factor, production_factor, node_type
      
    textinfo : string
      Textual info about the newly created dgraph and GT-Model
  """                  
      
  textinfo = ''
  
  dgraph = nx.erdos_renyi_graph(n, p, directed=True)
  
  # Remove possible multiedges and self loops  
  dgraph = nx.DiGraph(dgraph)
  dgraph.remove_edges_from(dgraph.selfloop_edges())
  
  # Set attributes for each node
  lmb                 = {}
  mu                  = {}
  consumption_factor  = {}
  production_factor   = {}
  node_type           = {}
  
  cash                = {}
  goods               = {}
  price               = {}
  
  for i in dgraph.nodes():
    lmb[i]                = lmb_i
    mu[i]                 = mu_i
    consumption_factor[i] = phi_i
    production_factor[i]  = pi_i
    node_type[i]          = node_type_i
    
    cash[i]               = c_ini
    goods[i]              = g_ini
    price[i]              = p_ini
  
  
  nx.set_node_attributes(dgraph, 'lmb', lmb)
  nx.set_node_attributes(dgraph, 'cash', cash)
  nx.set_node_attributes(dgraph, 'mu', mu)
  nx.set_node_attributes(dgraph, 'goods', goods)
  nx.set_node_attributes(dgraph, 'consumption_factor', consumption_factor)
  nx.set_node_attributes(dgraph, 'production_factor', production_factor)
  nx.set_node_attributes(dgraph, 'price', price)
  nx.set_node_attributes(dgraph, 'node_type', node_type)
    
  
  # As good links, the edge_type is always set to 1
  dgraph_tmp = dgraph.copy()
  dgraph.remove_edges_from(dgraph_tmp.edges())
  for edge in dgraph_tmp.edges():
    dgraph.add_edge(edge[0], edge[1], edge_type=1)
    
    
    
  # Add stats about the created erdos renyi
  sccs = nx.strongly_connected_components(dgraph)
  nscc = nx.number_strongly_connected_components(dgraph)
  ugraph = dgraph.to_undirected()
  ncc = nx.number_connected_components(ugraph)
  
  
  textinfo = 'Erdos Rényi directed network\n'
  textinfo += 'n='+str(n)+'\n'
  textinfo += 'p='+str(p)+'\n'
  textinfo += 'm='+str(dgraph.number_of_edges())+'\n'
  if(nx.is_directed_acyclic_graph(dgraph)):
    textinfo += 'Directed acyclic graph\n'
  else:
    textinfo += 'Directed cyclic graph\n'
  textinfo += 'scc='+str(nscc)+'\n'
  textinfo += 'ncc='+str(ncc)+'\n'
  textinfo += 'list of scc='+str(sccs)+'\n'
  textinfo += 'scc distribution='+str(sccpmf(dgraph))+'\n'
    
  return dgraph, textinfo





def create_gt_files(dgraph,
                    modelname,
                    submodelname,
                    minlayerid=0,
                    edgeextension='.edge0',
                    textinfo=''):
  """Create gt files from a networkx directed graph. The attributes of the directed graph
  are used to built resulting files. See below for more details on each mandatory attribute.
  
  Parameters
  ----
  dgraph : networkx directed graph
    The directed graph to transform to gt files
    The following node attributes are mandatory for each node:
    - lmb, cash
    - mu, goods, consumption_factor, production_factor
    - price
  modelname : string
    The name of the model = the name of the folder containing gt files
  submodelname : string
    Filename inside the folder. For instance if submodelname='erdos', the following
    files are created: erdos.vertex0, erdos.vertex1, erdos.vertex2, erdos.edge0
  minlayerid : int
    The starting layer id which corresponds to the cash layer.
    If minlayerid=0, the following files are created: erdos.vertex0, erdos.vertex1, erdos.vertex2
    If minlayerid=3, the following files are created: erdos.vertex3, erdos.vertex4, erdos.vertex5
  edgeextension : string
    The extension of edge file
  textinfo : string
    Additional text to add to statistic files

    
  Returns
  ----
  Files are created under folder/modelname which are:
  - vertex0, vertex1, vertex2 files (for minlayerid=0)
  - edge0 file (for edgeextension='.edge0')
  - _nodes_gt.csv file
  - _edges_gt.csv file
  - _stats_gt.txt file
    
  """
  
  # The default root folder is mcag.gtfolder. Change it if you would like to 
  # have gt-model files inside another folder
  updatemainrootfolder(mainrootfolder)
  folder = gtfolder
  
  directory = folder + modelname + '/'
  if not os.path.exists(directory):
    os.makedirs(directory)
  
  nodes = dgraph.nodes()
  edges = dgraph.edges()
  sccs  = nx.strongly_connected_components(dgraph)
  
  n     = dgraph.number_of_nodes()
  m     = dgraph.number_of_edges()
  nscc   = nx.number_strongly_connected_components(dgraph)
  
  ugraph = dgraph.to_undirected()
  ncc = nx.number_connected_components(ugraph)
  
  
  
  # Create GT vertices files
  # and at the same time creates gephi 
  lmb_n       = nx.get_node_attributes(dgraph,'lmb')
  cini_n      = nx.get_node_attributes(dgraph,'cash')
  mu_n        = nx.get_node_attributes(dgraph,'mu')
  gini_n      = nx.get_node_attributes(dgraph,'goods')
  phi_n       = nx.get_node_attributes(dgraph,'consumption_factor')
  pi_n        = nx.get_node_attributes(dgraph,'production_factor')
  pini_n      = nx.get_node_attributes(dgraph,'price')
  node_type_n = nx.get_node_attributes(dgraph,'node_type')
  
  vertexgephitext = ''
  vertexgephitext += 'Nodes Id Label node_type\n'
  for i in range(minlayerid, minlayerid+3, 1):
    vertextext = ''
    vertextext += str(n) + '\n'
    for nid in nodes:
      lmb         = lmb_n[nid]
      cini        = cini_n[nid]
      mu          = mu_n[nid]
      gini        = gini_n[nid]
      phi         = phi_n[nid]
      pi          = pi_n[nid]
      pini        = pini_n[nid]
      node_type   = node_type_n[nid]
      if i==minlayerid: # cash layer
        vertextext      += str(nid)+' '+str(lmb)+' '+str(cini)+'\n'
        vertexgephitext += str(nid)+' '+str(nid)+' '+str(nid)+' '+str(node_type)+'\n'
      if i==minlayerid+1: # goods layer
        vertextext += str(nid)+' '+str(mu)+' '+str(gini)+' '+str(phi)+' '+str(pi)+'\n'
      if i==minlayerid+2: # price layer
        vertextext += str(nid)+' '+str(pini)+' '+str(node_type)+'\n'
          
    vertexfullpath = directory + submodelname + '.vertex' + str(i)
    with open(vertexfullpath, 'w') as vertexfile:
      vertexfile.write(vertextext)
      vertexfile.close()
  
  vertexgephifullpath = directory + '_nodes_gt_' + submodelname + '.csv'
  with open(vertexgephifullpath, 'w') as vertexgephifile:
    vertexgephifile.write(vertexgephitext)
    vertexgephifile.close()
  
  
  
  # Create GT edges file
  # and create at the same time edge gephi
  edge_type_n = nx.get_edge_attributes(dgraph,'edge_type')
  
  edgetext = ''
  edgegephitext = ''
  edgetext      += str(m) + '\n'
  edgegephitext += 'Id Source Target edge_type\n'
  i = 0
  for edge in edges:
    edge_type = edge_type_n[edge]
    start     = edge[0]
    end       = edge[1]
    edgetext      += str(i)+' '+str(start)+' '+str(end)+'\n'
    edgegephitext += str(i)+' '+str(start)+' '+str(end)+' '+str(edge_type)+'\n'
    i = i+1
  
  edgefullpath = directory + submodelname + edgeextension
  with open(edgefullpath, 'w') as edgefile:
    edgefile.write(edgetext)
    edgefile.close()
  
  edgegephifullpath = directory + '_edges_gt_' + submodelname + '.csv'
  with open(edgegephifullpath, 'w') as edgegephifile:
    edgegephifile.write(edgegephitext)
    edgegephifile.close()
  
  
  

  """
  # Create GT vertices files
  # and at the same time creates gephi 
  vertexgephifullpath = directory + '_nodes_gt_' + submodelname + '.csv'
  with open(vertexgephifullpath, 'w') as vertexgephifile:
    vertexgephifile.write('Nodes Id Label node_type\n')
    for i in range(minlayerid, minlayerid+3, 1):
      vertexfullpath = directory + submodelname + '.vertex' + str(i)
      with open(vertexfullpath, 'w') as vertexfile:
        vertexfile.write(str(n) + '\n')
        for nid in nodes:
          lmb       = nx.get_node_attributes(dgraph,'lmb')[nid]
          cini      = nx.get_node_attributes(dgraph,'cash')[nid]
          mu        = nx.get_node_attributes(dgraph,'mu')[nid]
          gini      = nx.get_node_attributes(dgraph,'goods')[nid]
          phi       = nx.get_node_attributes(dgraph,'consumption_factor')[nid]
          pi        = nx.get_node_attributes(dgraph,'production_factor')[nid]
          pini      = nx.get_node_attributes(dgraph,'price')[nid]
          node_type = nx.get_node_attributes(dgraph,'node_type')[nid]
          if i==minlayerid: # cash layer
            vertexfile.write(str(nid)+' '+str(lmb)+' '+str(cini)+'\n')
            vertexgephifile.write(str(nid)+' '+str(nid)+' '+str(nid)+' '+str(node_type)+'\n')
          if i==minlayerid+1: # goods layer
            vertexfile.write(str(nid)+' '+str(mu)+' '+str(gini)+' '+str(phi)+' '+str(pi)+'\n')
          if i==minlayerid+2: # price layer
            vertexfile.write(str(nid)+' '+str(pini)+'\n')
      vertexfile.close()
  vertexgephifile.close()
    
  
  # Create GT edges file
  # and create at the same time edge gephi
  edgefullpath = directory + submodelname + edgeextension
  edgegephifullpath = directory + '_edges_gt_' + submodelname + '.csv'
  with open(edgefullpath, 'w') as edgefile:
    with open(edgegephifullpath, 'w') as edgegephifile:
      edgefile.write(str(m) + '\n')
      edgegephifile.write('Id Source Target edge_type\n')
      i = 0
      for edge in edges:
        edge_type = nx.get_edge_attributes(dgraph,'edge_type')[edge]
        start     = edge[0]
        end       = edge[1]
        edgefile.write(str(i)+' '+str(start)+' '+str(end)+'\n')
        edgegephifile.write(str(i)+' '+str(start)+' '+str(end)+' '+str(edge_type)+'\n')
        i = i+1
    edgegephifile.close()
  edgefile.close()
  """
  
  
  
    
  # Stats file
  statsfullpath = directory + '_stats_gt.txt'
  with open(statsfullpath, 'w') as statefile:
      
    statefile.write(textinfo+'\n')
    statefile.write('\n')
    statefile.write('n='+str(n)+'\n')
    statefile.write('m='+str(m)+'\n')
    if(nx.is_directed_acyclic_graph(dgraph)):
      statefile.write('Directed acyclic graph\n')
    else:
      statefile.write('Directed cyclic graph\n')
    statefile.write('scc='+str(nscc)+'\n')
    statefile.write('ncc='+str(ncc)+'\n')
    statefile.write('list of scc='+str(sccs)+'\n')
    statefile.write('scc distribution='+str(sccpmf(dgraph))+'\n')
  statefile.close()





def gt_files_to_dgraph(gtfilesfolder,
                       submodelname,
                       minlayerid=0,
                       edgeextension='.edge0'):
  """Crete a networkx directed graph with the corresponding GT-Mode attributes
  from GT files (by default edge0 and vertex0..2 files).
  This function is more complete.
  
  Parameters
  ----
  gtfilesfolder: string
    The path of the folder where GT files are stored (with trailing /). 
    
  submodelname: string
    The name of the edge0, vertex0..2 files without the extensions.
    
  minlayerid: int
    The default value is 0 meaning that vertex files start from 0 to 2
    
  edgeextension: string
    The default value is .edge0 meaning that the edge file has the extension .edge0
  
  Returns
  ----
  dgraph : A directed networkx graph corresponding to the GT files.
    Each node of this dgraph has the following attributes
      lmb, mu, cash, goods, price, consumption_factor, production_factor
  
  """
  
  lmb                 = {}
  mu                  = {}
  consumption_factor  = {}
  production_factor   = {}
  node_type           = {}
  
  cash                = {}
  goods               = {}
  price               = {}
  
  
  # Treat each vertex file with each node attribute
  for layer in [minlayerid, minlayerid+1, minlayerid+2]:
    vertexpath = gtfilesfolder + submodelname + '.vertex' + str(layer)
    
    with open(vertexpath, 'r') as vertexfile: 
      linenumber = 0
      for line in vertexfile:
        if not linenumber==0:
          line_array = re.split('\s+', line)
          
          nid_value = int(line_array[0])
          if layer==minlayerid:
            lmb[nid_value]                = float(line_array[1])
            cash[nid_value]               = float(line_array[2])
          if layer==minlayerid+1:
            mu[nid_value]                 = float(line_array[1])
            goods[nid_value]              = float(line_array[2])
            consumption_factor[nid_value] = float(line_array[3])
            production_factor[nid_value]  = float(line_array[4])
          if layer==minlayerid+2:
            price[nid_value]              = float(line_array[1])
            node_type[nid_value]          = int(line_array[2])

        linenumber += 1
    vertexfile.close()
  
  
  # Treat the edge file with edge_type attribute
  edge_type = {}
  edgepath = gtfilesfolder + submodelname + edgeextension
  with open(edgepath, 'r') as edgefile:
    linenumber = 0
    for line in edgefile:
      if not linenumber==0:
        line_array = re.split('\s+', line)
        start = int(line_array[1])
        end   = int(line_array[2])
        edge_type[(start,end)] = 1 # Goods link is always with type 1
      linenumber += 1
  edgefile.close()
  
  dgraph = nx.DiGraph()
  
  dgraph.add_nodes_from(list(lmb))
  nx.set_node_attributes(dgraph, 'lmb', lmb)
  nx.set_node_attributes(dgraph, 'cash', cash)
  nx.set_node_attributes(dgraph, 'mu', mu)
  nx.set_node_attributes(dgraph, 'goods', goods)
  nx.set_node_attributes(dgraph, 'consumption_factor', consumption_factor)
  nx.set_node_attributes(dgraph, 'production_factor', production_factor)
  nx.set_node_attributes(dgraph, 'price', price)
  nx.set_node_attributes(dgraph, 'node_type', node_type)
  
  dgraph.add_edges_from(list(edge_type))
  nx.set_edge_attributes(dgraph, 'edge_type', edge_type)
    
  return dgraph





def create_labour_files(dgraph,
                        modelname,
                        submodelname,
                        minlayerid=0,
                        edgeextension='.edge1',
                        textinfo='',
                        folder=lgmfolder):
  """ Create lgm layer files from networkx directed graph
  
  Parameters
  ----
  dgraph : networkx directed graph (corresponding to cash direction structure, not hour direction)
    The directed graph to transform to gt files
    The following node attributes are mandatory for each node:
    - psi, hour, hour_creation
    - gamma, cash
    - wage
  modelname : string
    The name of the model = the name of the folder containing lgm files
  submodelname : string
    Filename inside the folder. For instance if submodelname='erdos', the following
    files are created: erdos.vertex0, erdos.vertex1, erdos.vertex2, erdos.edge0
  minlayerid : int
    The starting layer id which corresponds to the hour layer.
    If minlayerid=0, the following files are created: erdos.vertex0, erdos.vertex1, erdos.vertex2
  edgeextension : string
    The extension of edge file (.edge1 by default)
  textinfo : string
    Additional text to add to statistic files
  folder : string
    Full path of where the lgm files will be created. The default path is mcag.lgmfolder path
  
  Returns
  ----
  
  """
  directory = folder + modelname + '/'
  if not os.path.exists(directory):
    os.makedirs(directory)
  
  nodes = dgraph.nodes()
  edges = dgraph.edges()
  sccs  = nx.strongly_connected_components(dgraph)
  
  n     = dgraph.number_of_nodes()
  m     = dgraph.number_of_edges()
  nscc   = nx.number_strongly_connected_components(dgraph)
  
  ugraph = dgraph.to_undirected()
  ncc = nx.number_connected_components(ugraph)
  
  psi_n           = nx.get_node_attributes(dgraph,'psi')
  hour_n          = nx.get_node_attributes(dgraph,'hour')
  hour_creation_n = nx.get_node_attributes(dgraph,'hour_creation')
  gamma_n         = nx.get_node_attributes(dgraph,'gamma')
  cini_n          = nx.get_node_attributes(dgraph,'cash')
  wini_n          = nx.get_node_attributes(dgraph,'wage')
  node_type_n     = nx.get_node_attributes(dgraph,'node_type')
  
  vertexgephitext = ''
  vertexgephitext += 'Nodes Id Label node_type\n'
  for i in range(minlayerid, minlayerid+3, 1):
    vertextext = ''
    vertextext += str(n) + '\n'
    for nid in sorted(nodes):
      psi           = psi_n[nid]
      hour          = hour_n[nid]
      hour_creation = hour_creation_n[nid]
      gamma         = gamma_n[nid]
      cini          = cini_n[nid]
      wini          = wini_n[nid]
      node_type     = node_type_n[nid]
      if i==minlayerid: # hour layer
        vertextext      += str(nid)+' '+str(psi)+' '+str(hour)+' '+str(hour_creation)+'\n'
        vertexgephitext += str(nid)+' '+str(nid)+' '+str(nid)+' '+str(node_type)+'\n'
      if i==minlayerid+1: # cash layer
        vertextext += str(nid)+' '+str(gamma)+' '+str(cini)+'\n'
      if i==minlayerid+2: # wage layer
        vertextext += str(nid)+' '+str(wini)+' '+str(node_type)+'\n'
        
    vertexfullpath = directory + submodelname + '.vertex' + str(i)
    with open(vertexfullpath, 'w') as vertexfile:
      vertexfile.write(vertextext)
      vertexfile.close()
      
  vertexgephifullpath = directory + '_nodes_lgm_' + submodelname + '.csv'
  with open(vertexgephifullpath, 'w') as vertexgephifile:
    vertexgephifile.write(vertexgephitext)
    vertexgephifile.close()
  
  
  edge_type_n = nx.get_edge_attributes(dgraph,'edge_type')
  
  edgetext = ''
  edgegephitext = ''
  edgetext      += str(m) + '\n'
  edgegephitext += 'Id Source Target edge_type\n'
  i = 0
  for edge in edges:
    edge_type = edge_type_n[edge]
    start     = edge[0]
    end       = edge[1]
    edgetext      += str(i)+' '+str(start)+' '+str(end)+'\n'
    edgegephitext += str(i)+' '+str(start)+' '+str(end)+' '+str(edge_type)+'\n'
    i = i+1
  
  edgefullpath = directory + submodelname + edgeextension
  with open(edgefullpath, 'w') as edgefile:
    edgefile.write(edgetext)
    edgefile.close()
  
  edgegephifullpath = directory + '_edges_lgm_' + submodelname + '.csv'
  with open(edgegephifullpath, 'w') as edgegephifile:
    edgegephifile.write(edgegephitext)
    edgegephifile.close()
  
  """
  
  # Create LGM vertices files
  # and at the same time creates gephi 
  #vertexgephifullpath = directory + '_nodes_lgm_' + submodelname + '.csv'
  #with open(vertexgephifullpath, 'w') as vertexgephifile:
    #vertexgephifile.write('Nodes Id Label node_type\n')
    vertexgephitext += 'Nodes Id Label node_type\n'
    for i in range(minlayerid, minlayerid+3, 1):
      vertexfullpath = directory + submodelname + '.vertex' + str(i)
      #with open(vertexfullpath, 'w') as vertexfile:
        #vertexfile.write(str(n) + '\n')
        vertextext += str(n) + '\n'
        for nid in nodes:
          psi           = nx.get_node_attributes(dgraph,'psi')[nid]
          hour          = nx.get_node_attributes(dgraph,'hour')[nid]
          hour_creation = nx.get_node_attributes(dgraph,'hour_creation')[nid]
          gamma         = nx.get_node_attributes(dgraph,'gamma')[nid]
          cini          = nx.get_node_attributes(dgraph,'cash')[nid]
          wini          = nx.get_node_attributes(dgraph,'wage')[nid]
          node_type     = nx.get_node_attributes(dgraph,'node_type')[nid]
          if i==minlayerid: # hour layer
            #vertexfile.write(str(nid)+' '+str(psi)+' '+str(hour)+' '+str(hour_creation)+'\n')
            #vertexgephifile.write(str(nid)+' '+str(nid)+' '+str(nid)+' '+str(node_type)+'\n')
            vertextext      += str(nid)+' '+str(psi)+' '+str(hour)+' '+str(hour_creation)+'\n'
            vertexgephitext += str(nid)+' '+str(nid)+' '+str(nid)+' '+str(node_type)+'\n'
          if i==minlayerid+1: # cash layer
            #vertexfile.write(str(nid)+' '+str(gamma)+' '+str(cini)+'\n')
            vertextext += str(nid)+' '+str(gamma)+' '+str(cini)+'\n'
          if i==minlayerid+2: # wage layer
            #vertexfile.write(str(nid)+' '+str(wini)+' '+str(node_type)+'\n')
            vertextext += str(nid)+' '+str(wini)+' '+str(node_type)+'\n'
      #vertexfile.close()
  #vertexgephifile.close()
    
  
  # Create LGM edges file
  # and create at the same time edge gephi
  edgefullpath = directory + submodelname + edgeextension
  edgegephifullpath = directory + '_edges_lgm_' + submodelname + '.csv'
  with open(edgefullpath, 'w') as edgefile:
    with open(edgegephifullpath, 'w') as edgegephifile:
      #edgefile.write(str(m) + '\n')
      #edgegephifile.write('Id Source Target edge_type\n')
      edgetext      += str(m) + '\n'
      edgegephitext += 'Id Source Target edge_type\n'
      i = 0
      for edge in edges:
        edge_type = nx.get_edge_attributes(dgraph,'edge_type')[edge]
        start     = edge[0]
        end       = edge[1]
        #edgefile.write(str(i)+' '+str(start)+' '+str(end)+'\n')
        #edgegephifile.write(str(i)+' '+str(start)+' '+str(end)+' '+str(edge_type)+'\n')
        edgetext      += str(i)+' '+str(start)+' '+str(end)+'\n'
        edgegephitext += str(i)+' '+str(start)+' '+str(end)+' '+str(edge_type)+'\n'
        i = i+1
    edgegephifile.close()
  edgefile.close()
  
  
  """

    
  # Stats file
  statsfullpath = directory + '_stats_lgm.txt'
  with open(statsfullpath, 'w') as statefile:
      
    statefile.write(textinfo+'\n')    
    statefile.write('\n')
    statefile.write('n='+str(n)+'\n')
    statefile.write('m='+str(m)+'\n')
    if(nx.is_directed_acyclic_graph(dgraph)):
      statefile.write('Directed acyclic graph\n')
    else:
      statefile.write('Directed cyclic graph\n')
    statefile.write('scc='+str(nscc)+'\n')
    statefile.write('ncc='+str(ncc)+'\n')
    statefile.write('list of scc='+str(sccs)+'\n')
    statefile.write('scc distribution='+str(sccpmf(dgraph))+'\n')
  statefile.close()
                    








#### TODO: Change this procedure to consider an erdos graph as networkx graph
def create_gt_erdos(n,
                    p,
                    modelname,
                    submodelname='erdos'):
  
  """Create directed erdos gt model with the below parameters and store it 
  to the corresponding directory.
  
  Parameters
  ----------
  n: order
  p: connection probability
  modelname: relative folder to gtfolder root
  submodelname: name of the files in each folder
  """
  
  dgraph = nx.erdos_renyi_graph(n, p, directed=True)
  
  # Remove possible multiedges and self loops  
  dgraph = nx.DiGraph(dgraph)
  dgraph.remove_edges_from(dgraph.selfloop_edges())
  
  textinfo   = 'Erdos random directed network\n'
  textinfo  += 'n='+str(n)+', p='+str(p)+'\n'
  
  create_gt_files(dgraph, modelname, submodelname, textinfo=textinfo)





  
  





def create_lgm_from_graphs(dgraph,
                           hini_e=8,
                           delta_f=0,
                           delta_e=8,
                           psi_f=0.1,
                           psi_e=0.9,
                           gamma_f=0.9,
                           gamma_e=0.1,
                     
                           lmb_f=0.1,
                           lmb_e=0.9,
                           mu_f=0.9,
                           mu_e=0.1,
                           pi_f=3,
                           pi_e=0,
                           phi_f=0.1,
                           phi_e=0.9):
  """Create lgm model from dgraph where the following attributes are already set
  for each node. It allows the manual creation of lgm model.
  cash, wage, goods, price, node_type (0:firm, 1:worker)
  each edge is also identified as edge_type (0:labour link, 1: goods link)
  
  This function does not remove the sink salary nodes, self loop and multiple edges.
  
  Parameters
  ----
  dgraph : networkx directed graph with set attributes
  
  Returns
  ----
  dgraph : networkx directed graph 
    Netowk corresponding to the whole topology with assignements of different 
    standard parameters for firms and workers.
  
  dgraph_labour : networkx directed graph
    Network corresponding to the labour topology
    
  dgraph_goods : networkx directed graph
    Network corresponding to the goods topology
    
  textinfo : string
    Textual info
  """
  
  textinfo = ''
  
  node_type = nx.get_node_attributes(dgraph,'node_type')
  fnodes    = [key for key in node_type if node_type[key]==0] # Get firm nodes
  enodes    = [key for key in node_type if node_type[key]==1] # Get worker nodes
  
  # Attributes for firms
  h                   = {} # Initial hour available
  delta               = {} # Hour created at each time iteration
  psi                 = {} # Amount of labour invested
  gamma               = {} # Amount of cash invested for paying salary
  
  lmb                 = {}
  mu                  = {}
  consumption_factor  = {}
  production_factor   = {}
  
  for i in fnodes:
    h[i]                  = 1e-6
    delta[i]              = delta_f
    psi[i]                = psi_f
    gamma[i]              = gamma_f
    
    lmb[i]                = lmb_f
    mu[i]                 = mu_f
    production_factor[i]  = pi_f
    consumption_factor[i] = phi_f

    
    
  # Attributes for workers
  for i in enodes:
    h[i]                  = hini_e
    delta[i]              = delta_e
    psi[i]                = psi_e
    gamma[i]              = gamma_e
  
    lmb[i]                = lmb_e
    mu[i]                 = mu_e
    production_factor[i]  = pi_e
    consumption_factor[i] = phi_e
    
  nx.set_node_attributes(dgraph, 'hour', h)
  nx.set_node_attributes(dgraph, 'hour_creation', delta)
  nx.set_node_attributes(dgraph, 'psi', psi)
  nx.set_node_attributes(dgraph, 'gamma', gamma)
    
  nx.set_node_attributes(dgraph, 'lmb', lmb)
  nx.set_node_attributes(dgraph, 'mu', mu)
  nx.set_node_attributes(dgraph, 'consumption_factor', consumption_factor)
  nx.set_node_attributes(dgraph, 'production_factor', production_factor)
  
  
  edge_type     = nx.get_edge_attributes(dgraph,'edge_type')
  labour_edges  = [key for key in edge_type if edge_type[key]==0] # labour links
  goods_edges   = [key for key in edge_type if edge_type[key]==1] # goods links
  
  # Split the initial graph to labour and goods directed graphs
  dgraph_labour = dgraph.copy()
  dgraph_labour.remove_edges_from(dgraph.edges())
  dgraph_labour.add_edges_from(labour_edges, edge_type=0)  

  dgraph_goods = dgraph.copy()
  dgraph_goods.remove_edges_from(dgraph.edges())
  dgraph_goods.add_edges_from(goods_edges, edge_type=1)  
    
  return dgraph, dgraph_labour, dgraph_goods, textinfo
  
  



def split_lgm_graph(dgraph):
  """This function splits the directed graph to labour and goods graphs
  
  Parameters
  ----
  dgraph : networkx digraph
    The directed graph to split
  
  Returns
  ----
  dgraph_labour : networkx digraph 
    The labour directed graph with only labour links (edge_type=0)
    
  dgraph_goods : networkx digraph
    The goods directed graph with only goods links (edge_type=1)
  """
  
  edge_type     = nx.get_edge_attributes(dgraph, 'edge_type')
  labour_edges  = [key for key in edge_type if edge_type[key]==0] # labour links
  goods_edges   = [key for key in edge_type if edge_type[key]==1] # goods links
  
  dgraph_labour = dgraph.copy()
  dgraph_labour.remove_edges_from(dgraph.edges())
  dgraph_labour.add_edges_from(labour_edges, edge_type=0)  

  dgraph_goods = dgraph.copy()
  dgraph_goods.remove_edges_from(dgraph.edges())
  dgraph_goods.add_edges_from(goods_edges, edge_type=1) 

  return dgraph_labour, dgraph_goods







def create_lgm_erdos_graphs(f,
                           e,
                           p,
                           removesinknodes=False,
                           hini_e=8,
                           cini_f=10000000,
                           cini_e=10000000,
                           wini_f=200,
                           wini_e=200,
                           pini_f=350,
                           pini_e=350,
                     
                           delta_f=0,
                           delta_e=8,
                           psi_f=0.1,
                           psi_e=0.9,
                           gamma_f=0.9,
                           gamma_e=0.1,
                     
                           lmb_f=0.1,
                           lmb_e=0.9,
                           mu_f=0.9,
                           mu_e=0.1,
                           pi_f=3,
                           pi_e=0,
                           phi_f=0.1,
                           phi_e=0.9):
                     
  """Create random graph for lgm model. The following attributes are returned:
  
  
  Such a model consists of two random bipartite graphs:
  - One having connection from firm to employees (labour connections)
  - One having connection from employees to firms (goods connections)
  
  Parameters
  ----------
  f : int
    number of firms
  e : int
    number of employees
  p : double
    the probability connection between f and e and vice versa
  ...
  
  Returns
  ----
  A triplet of directed graphs.
  dgraph : directed graph corresponding to the whole bipartite network
  dgraph_labour : directed graph corresponding to the labour network
  dgraph_goods : directed graph corresponding to the goods network
  
  """

  dgraph = nx.bipartite_random_graph(f,e,p,directed=True)
  
  # Remove possible multiedges and self loops  
  dgraph = nx.DiGraph(dgraph)
  dgraph.remove_edges_from(dgraph.selfloop_edges())
  
  
  
  
  # remove sink salary nodes (worker with kin!=0 and kout=0)
  if removesinknodes==True:
    fnodes, enodes = nx.algorithms.bipartite.sets(dgraph)
    for eid in enodes:
      e_kin   = dgraph.in_degree(eid)   # number of employers
      e_kout  = dgraph.out_degree(eid)  # number of sellers
      if e_kin!=0 and e_kout==0:
        dgraph.remove_node(eid)

  
  
  
  # Get some statistics from the whole directed graph
  sccs  = nx.strongly_connected_components(dgraph)
  
  n     = dgraph.number_of_nodes()
  m     = dgraph.number_of_edges()
  nscc  = nx.number_strongly_connected_components(dgraph)
  
  ugraph  = dgraph.to_undirected()
  ncc     = nx.number_connected_components(ugraph)
  
  
  fnodes, enodes = nx.algorithms.bipartite.sets(dgraph)
  nf = len(fnodes)
  ne = len(enodes)
  
  c = {}
  for i in fnodes:
    c[i] = cini_f
  for i in enodes:
    c[i] = cini_e
  nx.set_node_attributes(dgraph, 'cash', c)
  
  w = {}
  for i in fnodes:
    w[i] = wini_f
  for i in enodes:
    w[i] = wini_e
  nx.set_node_attributes(dgraph, 'wage', w)
  
  g = {}
  for i in dgraph.nodes():
    g[i] = 1e-6
  nx.set_node_attributes(dgraph, 'goods', g)
  
  price = {}
  for i in fnodes:
    price[i] = pini_f
  for i in enodes:
    price[i] = pini_e
  nx.set_node_attributes(dgraph, 'price', price)

  # Attributes for firms
  h                   = {} # Initial hour available
  delta               = {} # Hour created at each time iteration
  psi                 = {} # Amount of labour invested
  gamma               = {} # Amount of cash invested for paying salary
  
  lmb                 = {}
  mu                  = {}
  consumption_factor  = {}
  production_factor   = {}
  
  node_type           = {}
  for i in fnodes:
    h[i]                  = 1e-6
    delta[i]              = delta_f
    psi[i]                = psi_f
    gamma[i]              = gamma_f
    
    lmb[i]                = lmb_f
    mu[i]                 = mu_f
    production_factor[i]  = pi_f
    consumption_factor[i] = phi_f
    
    node_type[i]          = 0 # firm type
    
    
  # Attributes for employees
  for i in enodes:
    h[i]                  = hini_e
    delta[i]              = delta_e
    psi[i]                = psi_e
    gamma[i]              = gamma_e
  
    lmb[i]                = lmb_e
    mu[i]                 = mu_e
    production_factor[i]  = pi_e
    consumption_factor[i] = phi_e
    
    node_type[i]          = 1 # worker type
    
    
    
  nx.set_node_attributes(dgraph, 'hour', h)
  nx.set_node_attributes(dgraph, 'hour_creation', delta)
  nx.set_node_attributes(dgraph, 'psi', psi)
  nx.set_node_attributes(dgraph, 'gamma', gamma)
    
  nx.set_node_attributes(dgraph, 'lmb', lmb)
  nx.set_node_attributes(dgraph, 'mu', mu)
  nx.set_node_attributes(dgraph, 'consumption_factor', consumption_factor)
  nx.set_node_attributes(dgraph, 'production_factor', production_factor)
  
  nx.set_node_attributes(dgraph, 'node_type', node_type)
  
#   print(nx.get_node_attributes(dgraph,'cash'))
#   print(nx.get_node_attributes(dgraph,'wage'))
#   print(nx.get_node_attributes(dgraph,'goods'))
#   print(nx.get_node_attributes(dgraph,'price'))
#   
#   print('-----')
#   print('parameters')
#   
#   print('initial hour')
#  print(nx.get_node_attributes(dgraph,'hour'))
#   print('created hour in each iteration')
#   print(nx.get_node_attributes(dgraph,'hour_creation'))
#   print('hour investment')
#  print(nx.get_node_attributes(dgraph,'psi'))
#   print('salary investment')
#   print(nx.get_node_attributes(dgraph,'gamma'))
#   print('cash investment when buying goods')
#   print(nx.get_node_attributes(dgraph,'lmb'))
#   print('goods investment when selling goods')
#   print(nx.get_node_attributes(dgraph,'mu'))
#   print('consumption factor')
#   print(nx.get_node_attributes(dgraph,'consumption_factor'))
#   print('production factor')
#   print(nx.get_node_attributes(dgraph,'production_factor'))
  
  
  # Split the initial graph to labour and goods directed graphs
  dgraph_labour = dgraph.copy()
  dgraph_labour.remove_edges_from(dgraph.edges())
  for edge in dgraph.edges():
    if edge[0] in fnodes:
      dgraph_labour.add_edge(edge[0], edge[1], edge_type=0) # edge_type=0: labour link
  

  dgraph_goods = dgraph.copy()
  dgraph_goods.remove_edges_from(dgraph.edges())
  for edge in dgraph.edges():
    if edge[0] in enodes:
      dgraph_goods.add_edge(edge[0], edge[1], edge_type=1)  # edge_type=1: goods link

      
  textinfo   = 'Random LGM bipartite network\n'
  textinfo  += 'Stats (whole) - bipartite graph\n'
  textinfo  += '------------\n'
  textinfo  += 'n='+str(n)+', p='+str(p)+'\n'
  textinfo  += 'firms='+str(nf)+' (initial firm number='+str(f)+')\n'
  textinfo  += 'workers='+str(ne)+' (initial worker number='+str(e)+'\n'
  textinfo  += 'm='+str(m)+'\n'
  textinfo  += '\n'
  if(nx.is_directed_acyclic_graph(dgraph)):
    textinfo  += 'Directed acyclic whole network\n'
  else:
    textinfo  += 'Directed cyclic whole network\n'
  textinfo  += 'nscc='+str(nscc)+'\n'
  textinfo  += 'ncc='+str(ncc)+'\n'
  textinfo  += 'list of scc='+str(sccs)+'\n'
  textinfo  += 'scc distribution='+str(sccpmf(dgraph))+'\n'
  textinfo  += '------------\n'

  return dgraph, dgraph_labour, dgraph_goods, textinfo
  
  
  
  
  
  
  
def create_lgm_erdos(f,
                     e,
                     p,
                     modelname,
                     submodelname,
                     folder=lgmfolder,
                     removesinknodes=False):
  """Create random lgm model and store it inside a folder with the stats for each layer.
  The stats for whole network is stored inside _stats_lgm.txt file (1st part of this file)
  
  Parameters
  ----
  
  Returns
  ----
  All files necessary for the simulation of 
  
  """

  dgraph, dgraph_labour, dgraph_goods, textinfo = create_lgm_erdos_graphs(
    f, e, p, removesinknodes=removesinknodes, cini_e=1e-6, wini_e=0, pini_e=0
  )
  
#cini_e=1e-6 is important to avoid numerical errors

  create_labour_files(dgraph_labour,
                      modelname,
                      submodelname, 
                      minlayerid=0,
                      edgeextension='.edge1',
                      folder=folder,
                      textinfo=textinfo)

  create_gt_files(dgraph_goods, 
                  modelname, 
                  submodelname, 
                  minlayerid=3, 
                  edgeextension='.edge3', 
                  folder=folder)








def lgm_number_of_sensitive_labour_links(dgraph):
  """According to labour market context in directed bipartite graph,
  this function returns the number of sensitive labour links
  of a directed graph dgraph.
  This is a relative number normallized by the graph size+1 (avoid division by 0 error).
  """

  sens = len([n for (n,kin) in dgraph.in_degree_iter() if kin>=2])
  return sens/(dgraph.number_of_edges()+1)
  
  
def lgm_number_of_sensitive_goods_links(dgraph):
  """According to labour market context in directed bipartite graph,
  this function returns the number of sensitive labour links
  of a directed graph dgraph.
  This is a relative number normalized by the graph size+1 (avoid division by 0 error).
  """

  sens = len([n for (n,kout) in dgraph.in_degree_iter() if kout>=2])
  return sens/(dgraph.number_of_edges()+1)


def number_of_sink_nodes(dgraph):
  """Return the relative number of sink nodes in a directed graph.
  A sink node has only incoming neighbours.
  This is a relative number normalized by the graph order
  """
  kin = dgraph.in_degree()
  kout = dgraph.out_degree()
  sink = 0
  for i in sorted(kin):
    if kin[i]!=0 and kout[i]==0:
      sink = sink+1
  return sink / dgraph.number_of_nodes()
  
  

import xmlrpc.client as xml  
def view_graph(dgraph):
  server = xml.Server('http://127.0.0.1:20738/RPC2')
  G = server.ubigraph
  G.clear()
  
  for e in dgraph.edges():
    start = e[0]
    end = e[1]
    x = G.new_vertex()
    G.set_vertex_attribute(x, 'label', str(start))
    y = G.new_vertex()
    G.set_vertex_attribute(y, 'label', str(end))
    G.new_edge(x,y)
  
  




def gtfilestodigraph(modelname, nodefilename, edgefilename):
  """Return a networkx digraph from gt node and edge files
  
  Parameters
  ----
  - modelname: the name of the model referenced inside gtmodelinput folder
  - nodefilename: the exact filename where nodes are stored (e.g: erdos.edge0)
  The nodefile structure is:
  n
  nid x x x
  - edgefilename: the exact filename where edges are stored (e.g: erdos.vertex0)
  The edge file structure is:
  m
  mid start end
  
  Returns
  ----
  - a directed network digraph
  
  """

  node_fullpath = gtfolder+modelname+'/'+nodefilename
  edge_fullpath = gtfolder+modelname+'/'+edgefilename
  
  # Build the list of nodes
  
  nodelist = []
  with open(node_fullpath, 'r') as nodefile:
    i=0
    for line in nodefile:
      if not i==0:
        line_array = re.split('\s+', line) # '\s' is a regex for white space (tab, whitespace, newline)
        nid = int(line_array[0])
        nodelist.append(nid)
      i = i+1
    nodefile.close()
  
  # Build the list of edges
  edgelist = []
  with open(edge_fullpath, 'r') as edgefile:
    i=0
    for line in edgefile:
      if not i==0:
        line_array = re.split('\s+', line)
        start = int(line_array[1])
        end   = int(line_array[2])
        edgelist.append((start,end))
      i = i+1
    edgefile.close()
    
  # Build the digraph
  dgraph = nx.DiGraph()
  dgraph.add_nodes_from(nodelist)
  dgraph.add_edges_from(edgelist)
  
  return dgraph
  






def merge_with_random_graph(current_dgraph, p):
  """This function returns a directed graph which is the fusion between the edges
  of the current directed graph and a random directed graph.
  The edge doublons are not allowed.
  The orders of the current directed graph and the resulting directed graph are equal.
  The probability of connections between new distinct pairs of nodes is p.
  p_c is the probability connection between pairs of the current graph
  
  The resulting probability connection of new graph is (1-p_c)*p + p_c
  (linear estimation from p_c to 1)
  
  Parameters
  ----
  current_dgraph : NetworkX directed graph
  
  p: float
    Connection probability to build the resulting graph
  
  Returns
  ----
  new_dgraph: NetworkX directed graph
    The attributes of each node remain available but only number of edges changes
  """
  n   = current_dgraph.number_of_nodes()
  m_c = current_dgraph.number_of_edges()
  p_c = m_c / (n*(n-1))
  
  new_dgraph = current_dgraph
  
  edges = itertools.permutations(range(n),2)
  for e in edges:
    if random.random() < p:
      if not e in new_dgraph.edges():
        new_dgraph.add_edge(*e)
  

  return new_dgraph



