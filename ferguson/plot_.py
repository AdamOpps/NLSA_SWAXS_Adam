################################################################################
def plot(log_sigma,log_sum_kernel,x_linear,y_linear,sigma_opt,dimensionality,save_path:str=''):
# 
# copyright (c) Laura Williams & Russell Fung 2018
################################################################################
  
  try:
    # return without plotting if DO_NOT_PLOT is defined
    import os
    do_not_plot = os.environ['DO_NOT_PLOT']
    figure_name = 'ferguson.jpg'
    os.system('touch '+figure_name)
    return figure_name
  except:
    pass
  
  import matplotlib.pyplot as plt
  import numpy as np
  
  x = log_sigma
  y = log_sum_kernel
  y_min = np.min(y)
  y_max = np.max(y)
  
  x_sigma_opt = np.max(x_linear)
  y_sigma_opt = y_min
  x_dimensionality = x_sigma_opt
  y_dimensionality = 0.5*(y_min+y_max)
  
  plt.figure()
  plt.plot(x,y,'bo-',linewidth=2.0,fillstyle='none')
  plt.plot(x_linear,y_linear,'r-',linewidth=2.0)
  plt.xlabel(r'$\ln \sigma$',fontsize=15)
  plt.ylabel(r'$\ln \Sigma A_{ij}$',fontsize=15)
  plt.text(x_sigma_opt,y_sigma_opt,r'$\sigma_{opt}=%6.4f$'%sigma_opt,fontsize=15)
  plt.text(x_dimensionality,y_dimensionality,r'dimensionality=%6.2f'%dimensionality,fontsize=15)
  #plt.show(block=False)
  figure_name = 'ferguson.jpg'

  ## Append filename to specified save_path, if it exists
  if save_path:
    if save_path[-1] != os.sep:
      figure_name = os.sep + figure_name
    figure_name = save_path + figure_name

  plt.savefig(figure_name,bbox_inches='tight')
  plt.close()
  
  return figure_name

