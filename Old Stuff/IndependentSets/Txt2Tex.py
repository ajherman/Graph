import numpy as np
import sys

path = sys.argv[1]
X = np.loadtxt(path+'.txt',dtype=np.int32)
n = np.shape(X)[1]
s1 = "\\begin{tabular}{"+"l"*n+"} \n"
s2 = "\n \\end{tabluar}"
np.savetxt(path+"TEX.txt",X,delimiter = ' & ', newline='\\\  \n', fmt='%5d')
