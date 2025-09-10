###############################################################################
def histogram_int(z):
# 
# copyright (c) Russell Fung 2022
###############################################################################
  
  import numpy as np
  
  z_min = np.floor(np.min(z))
  z_max = np.ceil(np.max(z))
  num_bin = np.int_(z_max-z_min+1)
  my_bin = range(0,num_bin+1)+z_min-0.5
  bin_center = range(0,num_bin)+z_min
  hist,bins = np.histogram(z,bins=my_bin,density=False)
  
  return (hist,bin_center)

