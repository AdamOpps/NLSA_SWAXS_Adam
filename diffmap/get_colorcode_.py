################################################################################
def get_colorcode(N):
# 
# copyright (c) Russell Fung 2019
################################################################################
  
  from misc_tools import read_h5
  import numpy as np
  import os
  
  colorcode = np.arange(N)
  psi_fit = np.array([])
  if os.path.exists('colorcode.h5'):
    colorcode = read_h5('colorcode.h5','colorcode')
    try:
      psi_fit = read_h5('colorcode.h5','psi_fit')
    except:
      pass
  
  return colorcode,psi_fit

