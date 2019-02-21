import numpy as np

k,i = 17,16

def partitions(n,I=1):
    yield (n,)
    for i in range(I,n//2+1):
        for p in partitions(n-i,i):
            yield(i,)+p

def exponent(k,i):
    Pk = [x for x in partitions(k)]
    Pi = [x for x in partitions(i)]
    best_p = (k,)
    for p in Pk:
        if len(p)>len(best_p):
            if not True in [set(q)<=set(p) for q in Pi]:
                best_p = p
    return len(best_p),best_p

# Example
for k in range(3,30):
    print(k)
    for i in range(2,k):
        n,p=exponent(k,i)
        if n<k-i-1:
            print(True)
        elif n >k-i-1:
            print(False)
#
#            print("An asymptotic lower bound on the independence number of J(v,"+str(k)+","+str(i)+") is v^"+str(n))
#            print("Block sizes: ",set(p))

