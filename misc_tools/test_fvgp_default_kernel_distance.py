import numpy as np
from scipy.spatial.distance import cdist

x1 = np.random.uniform(size=(4,3))
x2 = np.random.uniform(size=(6,3))
dist1 = np.zeros((len(x1),len(x2)))
for i in range(len(x1[0])):
  dist1 += abs(np.subtract.outer(x1[:,i],x2[:,i]))**2
dist1 = np.sqrt(dist1)
dist2 = cdist(x1,x2,'euclidean')
error = dist1-dist2
print('x1 =')
print(x1)
print('x2 =')
print(x2)
print('default_kernel,distance:')
print(dist1)
print('cdist:')
print(dist2)
print('error:')
print(np.max(np.abs(error.reshape(-1,1))))
