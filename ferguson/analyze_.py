################################################################################
def analyze(yRow_yCol_yVal_file):
# 
# copyright (c) Laura Williams & Russell Fung 2018
################################################################################
  
  from .A_ij_              import A_ij
  from .fit_ramp_          import fit_ramp
  from .plot_              import plot
  from .sigma_of_interest_ import sigma_of_interest
  
  from misc_tools import read_h5
  import numpy as np
  
  Dsq = read_h5(yRow_yCol_yVal_file,'yVal')
  N = np.max(read_h5(yRow_yCol_yVal_file,'yRow'))
  sigma = sigma_of_interest(Dsq)
  num_sigma = len(sigma)
  
  # less memory required if for loop is used
  #A = A_ij(Dsq,sigma)
  A = np.zeros(num_sigma)
  for k in range(num_sigma):
    A[k] = A_ij(Dsq,sigma[k])
  
  tol = 0.05*np.log(N)
  p = 90
  
  x = np.log(sigma)
  y = np.log(A)
  xl,yl,x_mid,y_mid,dimensionality = fit_ramp(x,y,tol,p)
  sigma_opt = np.exp(x_mid)
  plot(x,y,xl,yl,sigma_opt,dimensionality)
  
  return sigma_opt,dimensionality

