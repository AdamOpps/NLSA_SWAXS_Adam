################################################################################
def plot_eigVal(eigVec_eigVal_file):
# 
# copyright (c) Russell Fung 2019
################################################################################
  
  try:
    # return without plotting if DO_NOT_PLOT is defined
    import os
    do_not_plot = os.environ['DO_NOT_PLOT']
    figure_name = 'eigen_spectrum.jpg'
    os.system('touch '+figure_name)
    return figure_name
  except:
    pass
  
  from misc_tools import read_h5
  
  eigVal = read_h5(eigVec_eigVal_file,'eigVal')
  
  import numpy as np
  # np.float deprecated & replaced with np.float64.
  eps = np.finfo(np.float64).eps
  
  eigVal = eigVal[1:]
  eigVal[np.where(np.abs(1-eigVal)<1.0e-6)[0]] = 1-eps
  
  import matplotlib.pyplot as plt
  
  nEigs = len(eigVal)
  x = range(1,nEigs+1)
  y = (1-eigVal[0])/(1-eigVal)
  
  fig = plt.figure()
  fig.set_size_inches(7,5)
  plt.plot(x,y,'bo-',linewidth=2.0,fillstyle='none')
  plt.xticks(x)
  plt.xlabel('spectral component#',fontsize=15)
  plt.ylabel('relative eigenvalue',fontsize=15)
  plt.show(block=False)
  figure_name = 'eigen_spectrum.jpg'
  plt.savefig(figure_name,bbox_inches='tight')
  plt.close()
  
  return figure_name

