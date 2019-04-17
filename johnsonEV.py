import numpy as np
from numpy import linalg
import itertools as it
from GraphFun import *


def choose(n,k):
	return int(np.prod([(n-i+1)/(i) for i in range(1,k+1)]))

def johnsonEV(v,k):
	return [(k-j)*(v-k-j)-j for j in range(0,min(k,v-k)+1)]

v,k,i=10,4,3
print("v: ",v,"k: ",k,"i: ", i)
verts=choose(v,k)
print("vertices: ",verts)
deg=k*(v-k)
print("valency: ",deg)

print("eigenvalues:")

m=np.zeros([verts,verts])
Gadj=genJohnsonAdjList(v,k,i,)
print(np.shape(m))
for j in range(verts):
	for s in Gadj[1][j]:
		m[j,s]=1


if i==k:
	jev=johnsonEV(v,k)
	minev=min(jev)
	print(jev,minev)
	print("simple bound: ")
	print(verts/v," <=alpha(J(v,k))<= ", choose(v,k)/(v-k+1))
else:
	jev=linalg.eig(m)
	minev=min(jev[0])
	print(minev)




print("eigenvalue ratio bound:")
print("alpha(J(v,k))<= ", verts/(1-deg/minev) )
