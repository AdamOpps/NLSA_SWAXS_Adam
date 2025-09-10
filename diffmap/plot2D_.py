################################################################################
def plot2D(eigVec_eigVal_file,psi_of_interest,cmap='gist_rainbow',s=10,show_colorbar=True):
# 
# copyright (c) Russell Fung 2019
################################################################################
  
  try:
    # return without plotting if DO_NOT_PLOT is defined
    import os
    do_not_plot = os.environ['DO_NOT_PLOT']
    figure_name = 'diffmap_2D.jpg'
    os.system('touch '+figure_name)
    return figure_name
  except:
    pass
  
  from .get_colorcode_ import get_colorcode
  
  from misc_tools import read_h5
  
  eigVec = read_h5(eigVec_eigVal_file,'eigVec')
  
  num_psi = len(psi_of_interest)
  x = eigVec[:,psi_of_interest[0]]/eigVec[:,0]
  y = eigVec[:,psi_of_interest[1]]/eigVec[:,0]
  my_xlabel = '$\Psi_'+str(psi_of_interest[0])+'$'
  my_ylabel = '$\Psi_'+str(psi_of_interest[1])+'$'
  if (num_psi>2):
    z = eigVec[:,psi_of_interest[2]]/eigVec[:,0]
    my_zlabel = '$\Psi_'+str(psi_of_interest[2])+'$'
  
  import matplotlib.pyplot as plt
  import numpy as np
  
  colorcode,psi_fit = get_colorcode(len(x))
  
  fig,ax1 = plt.subplots(1,1)
  fig.set_size_inches(8,4)
  if (num_psi>2):
    fig,(ax1,ax2,ax3) = plt.subplots(1,3)
    fig.set_size_inches(16,4)
  sc1 = ax1.scatter(x,y,c=colorcode,cmap=cmap,s=s)
  ax1.set_xlabel(my_xlabel,fontsize=15)
  ax1.set_ylabel(my_ylabel,fontsize=15)
  ax1.set_aspect('equal','box')
  if (num_psi>2):
    sc2 = ax2.scatter(x,z,c=colorcode,cmap=cmap,s=s)
    ax2.set_xlabel(my_xlabel,fontsize=15)
    ax2.set_ylabel(my_zlabel,fontsize=15)
    ax2.set_aspect('equal','box')
    sc3 = ax3.scatter(y,z,c=colorcode,cmap=cmap,s=s)
    ax3.set_xlabel(my_ylabel,fontsize=15)
    ax3.set_ylabel(my_zlabel,fontsize=15)
    ax3.set_aspect('equal','box')
  if show_colorbar:
    if (num_psi>2):
      plt.colorbar(sc3,ax=ax3)
    else:
      plt.colorbar(sc1,ax=ax1)
  if psi_fit.any():
    ax1.plot(psi_fit[:,0],psi_fit[:,1],'k-',linewidth=5,alpha=0.3)
    if (num_psi>2):
      ax2.plot(psi_fit[:,0],psi_fit[:,2],'k-',linewidth=5,alpha=0.3)
      ax3.plot(psi_fit[:,1],psi_fit[:,2],'k-',linewidth=5,alpha=0.3)
  plt.show(block=False)
  
  figure_name = 'diffmap_2D.jpg'
  plt.savefig(figure_name,bbox_inches='tight')
  plt.close()
  
  return figure_name

