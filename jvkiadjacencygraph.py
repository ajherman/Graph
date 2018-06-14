# -*- coding: utf-8 -*-
"""
Created on Mon May 21 17:27:02 2018

@author: Taiyo
"""

from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import networkx.generators.directed
import itertools as it


Jv='12345'
Jk=2
Ji=0

combos=list(it.combinations(Jv, Jk))



edges=[]
combos=[''.join(t[0] for t in x) for x in combos]

for i,x in enumerate(combos):
    for y in combos[i:]:
        if len(set(x) & set(y))==Ji:
            edges.append((x,y))
            edges.append((y,x))


G = nx.empty_graph(0, create_using=nx.DiGraph())

G.add_edges_from(e for e in edges)

A=nx.to_numpy_matrix(G)
print(A)
#
#
#pos = nx.layout.spectral_layout(G)
pos = nx.layout.shell_layout(G,[['45'],['12', '13', '23'],[ '14', '25', '34', '15', '24', '35']])
#pos = nx.layout.spring_layout(G,pos=nx.circular_layout(G),iterations=2)

#pos2 = {a[0]:a[1]+array([.2,.2] for a in pos}

node_sizes = [3 + 10 * i for i in range(len(G))]
M = G.number_of_edges()
edge_colors = [i for i in range(2, M + 2)]
#edge_alphas = [(5 + i) / (M + 4) for i in range(M)]

#nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='blue')
edges = nx.draw_networkx_edges(G, pos, node_size=node_sizes, arrowstyle='->',
                               arrowsize=10, edge_color=edge_colors,
                               edge_cmap=plt.cm.Blues, width=2)
nx.draw_networkx_labels(G,pos)
# set alpha value for each edge
#for i in range(M):
#    edges[i].set_alpha(1)

plt.figure(1,figsize=(12,12)) 
ax = plt.gca()
ax.set_axis_off()
plt.show()