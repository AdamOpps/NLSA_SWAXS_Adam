################################################################################
def plot1D(eigVec_eigVal_file,psi_of_interest):
# 
# copyright (c) Shanghui Huang & Russell Fung 2024
################################################################################
  
  try:
    # return without plotting if DO_NOT_PLOT is defined
    import os
    do_not_plot = os.environ['DO_NOT_PLOT']
    figure_name = 'eigVec_1D.jpg'
    os.system('touch '+figure_name)
    return figure_name
  except:
    pass
  
  from misc_tools import read_h5
  
  eigVec = read_h5(eigVec_eigVal_file,'eigVec')
  
  y1 = eigVec[:,psi_of_interest[0]]/eigVec[:,0]
  y2 = eigVec[:,psi_of_interest[1]]/eigVec[:,0]
  y3 = eigVec[:,psi_of_interest[2]]/eigVec[:,0]
  my_y1label = 'Eigenvector '+str(psi_of_interest[0])
  my_y2label = 'Eigenvector '+str(psi_of_interest[1])
  my_y3label = 'Eigenvector '+str(psi_of_interest[2])
  
  import matplotlib.pyplot as plt
  import numpy as np
  
  fig = plt.figure()
  ax1 = fig.add_subplot(311)
  ax2 = fig.add_subplot(312)
  ax3 = fig.add_subplot(313)
  fig.set_size_inches(6,3)
  ax1.plot(y1,label=my_y1label)
  ax1.legend()
  ax2.plot(y2,label=my_y2label)
  ax2.legend()
  ax3.plot(y3,label=my_y3label)
  ax3.legend()
  ax1.get_xaxis().set_visible(False)
  ax2.get_xaxis().set_visible(False)
  plt.show(block=False)
  
  figure_name = 'eigVec_1D.jpg'
  plt.savefig(figure_name,bbox_inches='tight')
  plt.close()
  
  return figure_name

