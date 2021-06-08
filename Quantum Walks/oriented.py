import numpy as np
import scipy.linalg as la

def is_int(x,tol):
    return np.all(np.abs(x-np.round(x))<tol)

for n in range(5,100): # Iterate over roots of unity
    titled=0 # Flag for whether section has been given a header

    # Matrix of sines
    roots=np.array([[2*np.sin(2*np.pi*i*j/n) for i in range(n)] for j in range(n)])

    # Build list of indices
    idx = [(i,) for i in range(n) if 4*i%n!=0]

    # Test
    for k in range(10**30):
        try:
            multi=idx[k]
            for i in range(multi[-1]): # CHANGED BY REMOVING +1
                if -i%n not in multi and 4*i%n!=0:
                    idx.append(multi+(i,))
        except:
            break

    for s in idx:
        val=np.sum(roots[s,:],axis=0)
        if is_int(val,1e-6):
            # Add header if there is at least one example for given n
            if not titled:
                print("============================================================")
                print("n = "+str(n))
                titled=1
            print(s)
