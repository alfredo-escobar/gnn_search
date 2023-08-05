import numpy as np

a = np.array([1,1,1,2,3])
b = np.array([1,2,3,3,3])

ab = np.array([[1,1,1,2,3],
               [1,2,3,3,3]])

c = np.zeros((5,5), dtype='float32')

d = b[-1:-5:-1]

e = np.array([7,4,8,5,2,6,7,8,4,5,6,1,9])

f = e[[5,12]]

c[ab] = 1
print(f)