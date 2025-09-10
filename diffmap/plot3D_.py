################################################################################
def plot3D(eigVec_eigVal_file,psi_of_interest,cmap='gist_rainbow',s=10,show_colorbar=True):
# 
# copyright (c) Shanghui Huang & Russell Fung 2024
################################################################################
  
  try:
    # return without plotting if DO_NOT_PLOT is defined
    import os
    do_not_plot = os.environ['DO_NOT_PLOT']
    figure_name = 'diffmap_3D.jpg'
    os.system('touch '+figure_name)
    return figure_name
  except:
    pass
  
  from .get_colorcode_ import get_colorcode
  
  from misc_tools import read_h5
  
  eigVec = read_h5(eigVec_eigVal_file,'eigVec')
  
  x = eigVec[:,psi_of_interest[0]]/eigVec[:,0]
  y = eigVec[:,psi_of_interest[1]]/eigVec[:,0]
  my_xlabel = '$\Psi_'+str(psi_of_interest[0])+'$'
  my_ylabel = '$\Psi_'+str(psi_of_interest[1])+'$'
  z = eigVec[:,psi_of_interest[2]]/eigVec[:,0]
  my_zlabel = '$\Psi_'+str(psi_of_interest[2])+'$'
  
  import matplotlib.pyplot as plt
  import numpy as np
  
  colorcode,psi_fit = get_colorcode(len(x))
  
  fig = plt.figure()
  ax = fig.add_subplot(projection='3d')
  fig.set_size_inches(7,5)
  sc = ax.scatter(x,y,z,c=colorcode,cmap=cmap,s=s)
  ax.set_xlabel(my_xlabel,fontsize=15)
  ax.set_ylabel(my_ylabel,fontsize=15)
  ax.set_zlabel(my_zlabel,fontsize=15)
  ax.set_aspect('auto','box')
  if show_colorbar:
    plt.colorbar(sc,ax=ax)
  if psi_fit.any():
    ax.plot(psi_fit[:,0],psi_fit[:,1],psi_fit[:,2],'k-',linewidth=5,alpha=0.3)
  plt.show(block=False)
  
  figure_name = 'diffmap_3D.jpg'
  plt.savefig(figure_name,bbox_inches='tight')
  plt.close()
  
  return figure_name

