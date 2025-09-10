################################################################################
def normalize(yRow,yCol,yVal,alpha):
# 
# copyright (c) Laura Williams & Russell Fung 2018
# updated April 11, 2022
################################################################################
  
  from misc_tools import write_h5
  import numpy as np
  from scipy.sparse import coo_matrix
  import os
  
  numRow = np.max(yRow)+1
  
  K = coo_matrix((yVal,(yRow,yCol)),shape=(numRow,numRow))
  row_sum = np.array(np.sum(K,axis=1))
  row_sum = row_sum.flatten()
  del K
  
  normalization = np.power(row_sum[yRow]*row_sum[yCol],-alpha)
  yVal = yVal*normalization
  
  W = coo_matrix((yVal,(yRow,yCol)),shape=(numRow,numRow))
  D = np.array(np.sum(W,axis=1))
  D = D.flatten()
  del W
  
  normalization = np.power(D[yRow]*D[yCol],-0.5)
  yVal = yVal*normalization
  
  h5_name = 'column_sum_and_D.h5'
  try:
    os.remove(h5_name)
  except:
    pass
  write_h5(h5_name,row_sum,'column_sum')
  write_h5(h5_name,D,'column_D')
  
  return yVal
  
