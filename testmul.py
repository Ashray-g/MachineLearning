import numpy as np
import time

X = np.random.rand(4000, 4000)
Y = np.random.rand(4000, 1)

s = time.perf_counter()
Z = X @ Y
t1 = time.perf_counter() - s

s = time.perf_counter()

Z2 = np.empty((4000, 1))
for i in range(4000):
    sum = 0
    for j in range(4000):
        sum += X[i, j] * Y[j, 0]

    Z2[i, 0] = sum   
t2 = time.perf_counter() - s


print("np", t1)
print("norm", t2)
print("faster", t2/t1, "x")