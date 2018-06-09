# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 15:14:19 2018

@author: TEA
"""

# Finding independent sets with Boltzmann machines

import numpy as np
import random
import networkx as nx
import networkx.algorithms.approximation as nxaa
import matplotlib.pyplot as plt
import networkx.generators.directed
import itertools as it
import time as tm
from GraphFun import *

dtype = "int32"
niters = 100 # Number of iterations
ntrials = 50

##################
# Adjacency matrix
##################

# Jvki
v,k,i = 5,2,0
G = genJohnsonGraph(v,k,i)
A = getAdjArray(G)

##################################
# Find Fractional Chromatic Number
##################################
tic = tm.time()
chi = getFracChromNumber(A,niters,ntrials)
toc = tm.time()

# Print independent set
print("Run time: "+str(toc-tic)+"s")
print("Fractional chromatic number computed by stochastic algorithm: "+str(chi))
