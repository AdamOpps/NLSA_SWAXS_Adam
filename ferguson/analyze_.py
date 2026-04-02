################################################################################
def analyze(yRow_yCol_yVal_file, verbose:bool=False, plot_save_path:str=''):
# 
# copyright (c) Laura Williams & Russell Fung 2018
# updated April 2026 by Adam Opperman
################################################################################
  
  '''
    Calculates the optimal length scale, sigma, given the input squared Euclidean distances.
    Parameters:
      - yRow_yCol_yVal_file (String or Path-like) \n
            Filename (and path) of the squared Euclidean distance matrix .h5 file
      - verbose (Bool, default=False) \n
            False: returns  sigma_opt, dimensionality \n
            True: returns data used to generate the "Ferguson Plot":
                sigma_opt, dimensionality, log(A), log(sigmas), line_fit_x, line_fit_y
      - plot_save_path (String, default='') \n
            Path to specify where the default "ferguson.jpg" plot is saved. \n
            Default: saves to the current working directory.
  '''

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
  
  # OLD: Treat sigma as array; calculate A vectorized
  #A = A_ij(Dsq,sigma)

  # NEW: Calculate for one sigma at a time
  # less memory required if for loop is used
  A = np.zeros(num_sigma)
  for k in range(num_sigma):
    A_calced = A_ij(Dsq,sigma[k])
    A[k] = A_calced

    # # A_ij() may return a zero-dimensional array; manually convert to scalar for safety
    # try:
    #   if isinstance(A_calced, np.ndarray): A[k] = A_calced.item()
    #   else: A[k] = A_calced
    # # Handle errors and provide info.
    # except ValueError as e:
    #   print(
    #     '''
    #     Error encountered in "./ferguson/analyze_.py": 
    #     -> Unable to set element of array A with result of A_ij().
    #     \tData Type: %s
    #     \tSize: %s
    #     \tContains NaN: %s
    #     ''' % (type(A_calced), np.size(A_calced), np.isnan(A_calced).sum()>0)
    #   )
    #   print(e)
    #   break
  
  tol = 0.05*np.log(N)
  p = 90
  
  x = np.log(sigma)
  y = np.log(A)
  xl,yl,x_mid,y_mid,dimensionality = fit_ramp(x,y,tol,p)
  sigma_opt = np.exp(x_mid)
  plot(x,y,xl,yl,sigma_opt,dimensionality,save_path=plot_save_path)
  
  if verbose:
    # return log(A), log(sigma), and linear fit data for manual plotting elsewhere 
    return sigma_opt,dimensionality,x,y,xl,yl
  else:
    # DEFAULT: Return optimal sigma and dimensionality
    return sigma_opt,dimensionality

