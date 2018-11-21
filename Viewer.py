# Save txt file showing a specific slice of values

import numpy as np
from GraphFun import *

# Load 3d array 
alphas = np.load("IndependentSets/JvkiAlphas.npy")

## Johnson Graphs
#display_alphas = alphas[sliceIdx('johnson',21)]
#np.savetxt("IndependentSets/JohnsonAlphas.txt",display_alphas,fmt='%5d')

## Fixed v graphs
#vv = 15
#display_alphas = alphas[sliceIdx('fixedv',vv)]
#np.savetxt("IndependentSets/J"+str(vv)+"kiAlphas.txt",display_alphas,fmt='%5d',newline='\r\n')

### J(v,4,2)
#display_alphas = alphas[[v for v in range(23)],23*[4],23*[2]]
#np.savetxt("IndependentSets/Jv42Alphas.txt",display_alphas,fmt='%5d',newline='\r\n')

### J(v,4,1)
#display_alphas = alphas[[v for v in range(30)],30*[4],30*[1]]
#np.savetxt("IndependentSets/Jv41Alphas.txt",display_alphas,fmt='%5d',newline='\r\n')

### J(v,5,1)
#display_alphas = alphas[[v for v in range(30)],30*[5],30*[1]]
#np.savetxt("IndependentSets/Jv51Alphas.txt",display_alphas,fmt='%5d',newline='\r\n')

### J(v,6,1)
#display_alphas = alphas[[v for v in range(30)],30*[6],30*[1]]
#np.savetxt("IndependentSets/Jv61Alphas.txt",display_alphas,fmt='%5d',newline='\r\n')

## J(v,7,1)
display_alphas = alphas[[v for v in range(30)],30*[7],30*[1]]
np.savetxt("IndependentSets/Jv71Alphas.txt",display_alphas,fmt='%5d',newline='\r\n')

### J(v,8,1)
#display_alphas = alphas[[v for v in range(30)],30*[8],30*[1]]
#np.savetxt("IndependentSets/Jv81Alphas.txt",display_alphas,fmt='%5d',newline='\r\n')


# Fixed i graphs
ii = 1
display_alphas = alphas[sliceIdx('kneser',21,s=ii)]
np.savetxt("IndependentSets/Jvk"+str(ii)+"Alphas.txt",display_alphas,fmt='%5d',newline='\r\n')
