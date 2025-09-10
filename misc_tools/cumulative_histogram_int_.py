###############################################################################
def cumulative_histogram_int(z):
# 
# copyright (c) Russell Fung 2022
###############################################################################
  
  from .histogram_int_ import histogram_int
  import numpy as np
  
  hist,bin_center = histogram_int(z)
  hist = np.cumsum(hist)
  
  return (hist,bin_center)

