
import numpy as np
import example
import time
from numba import vectorize,float64


z = np.random.rand(3,3)
z = np.zeros((300,3000)) + 3.
#y = np.random.rand(3,3)
y = np.zeros((3000,300)) + 4.
#res = np.zeros((3, 3))
#print(z)
#print(y)
start = time.time()
t = example.multip_func(z,y)
end = time.time()
print("cython",(end-start) * 10**3, 'ms')
start = time.time()
a = z @ y
end = time.time()
print("numpy",(end-start) * 10**3, 'ms')
#print(t)
#print(a)

