from fvgp import gp
import numpy as np
from scipy.spatial.distance import cdist

x1 = np.random.uniform(size=(4,3))
x2 = np.random.uniform(size=(6,3))
dist1 = gp.GP._get_distance_matrix(None,x1,x2)
dist2 = cdist(x1,x2,'euclidean')
error = dist1-dist2
print('x1 =')
print(x1)
print('x2 =')
print(x2)
print('get_distance_matrix:')
print(dist1)
print('cdist:')
print(dist2)
print('error:')
print(np.max(np.abs(error.reshape(-1,1))))
