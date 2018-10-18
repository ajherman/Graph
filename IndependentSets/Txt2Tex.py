import numpy as np
import sys

path = "Jvk1Alphas" #sys.argv[0]
X = np.loadtxt(path+'.txt',dtype=np.int32)
np.savetxt(path+"TEX.txt",X,delimiter = ' & ', newline='\\\  \n', fmt='%5d')
