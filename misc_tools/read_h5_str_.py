def read_h5_str(filename,variable,h5=True):
# 
# copyright (c) Russell Fung 2024
################################################################################
  
  if h5:
    import h5py
    import numpy as np
    
    f = h5py.File(filename,'r')
    x = np.array(f[variable])
    x = str(x)
    if x[0]=='b': x = x[2:-1]
  else:
    from scipy.io import loadmat
    
    f = loadmat(filename)
    x = f[variable]
  
  return x

