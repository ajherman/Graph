import math

x = input("Enter v,k,i (separated by spaces): ")

v,k,i = x.split()
v,k,i = int(v),int(k),int(i)

n_vertices=math.comb(v,k)
degree = math.comb(k,i)*math.comb(v-k,k-i)
n_edges=n_vertices*degree//2
n_possible_edges=math.comb(n_vertices,2)

density = n_edges/n_possible_edges

print("Density: ",density)