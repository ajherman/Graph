# Save txt file showing a specific slice of values

import numpy as np
from GraphFun import *

# Load 3d array 
alphas = np.load("IndependentSets/JvkiAlphas.npy")

## Johnson Graphs
#display_alphas = alphas[sliceIdx('johnson',21)]
#np.savetxt("IndependentSets/JohnsonAlphas.txt",display_alphas,fmt='%5d')

# Fixed v graphs
vv = 15
display_alphas = alphas[sliceIdx('fixedv',vv)]
np.savetxt("IndependentSets/J"+str(vv)+"kiAlphas.txt",display_alphas,fmt='%5d',newline='\r\n')

## Fixed i graphs
#ii = 1
#display_alphas = alphas[sliceIdx('kneser',21,s=ii)]
#np.savetxt("IndependentSets/Jvk"+str(ii)+"Alphas.txt",display_alphas,fmt='%5d',newline='\r\n')
