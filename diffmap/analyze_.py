################################################################################
def analyze(yRow_yCol_yVal_file,sigma,nEigs,alpha):
# 
# copyright (c) Laura Williams & Russell Fung 2018
################################################################################
  
  from .normalize_ import normalize
  
  from misc_tools import read_h5,write_h5
  import numpy as np
  from scipy.sparse import coo_matrix,linalg
  import os
  
  yRow = read_h5(yRow_yCol_yVal_file,'yRow')
  yCol = read_h5(yRow_yCol_yVal_file,'yCol')
  yVal = read_h5(yRow_yCol_yVal_file,'yVal')
  
  yRow = yRow[0,:]-1
  yCol = yCol[0,:]-1
  numRow = np.max(yRow)+1
  # this is K
  yVal = np.exp(-yVal[0,:]/sigma/sigma)
  
  # this is L
  yVal = normalize(yRow,yCol,yVal,alpha)
  L = coo_matrix((yVal,(yRow,yCol)),shape=(numRow,numRow))
  
  eigVal,eigVec = linalg.eigsh(L,k=nEigs+1,which='LA',maxiter=200)
  eigVal = np.flip(eigVal,axis=0)
  eigVec = np.fliplr(eigVec)
  
  eigVec_eigVal_file = 'dataPsi_sigma%.4f'%sigma+'.h5'
  try:
    os.remove(eigVec_eigVal_file)
  except:
    pass
  write_h5(eigVec_eigVal_file,eigVal,'eigVal')
  write_h5(eigVec_eigVal_file,eigVec,'eigVec')
  return eigVec_eigVal_file
 
