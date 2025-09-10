################################################################################
def symmetrize(yRow,yCol,yVal):
# 
# copyright (c) Russell Fung 2020
################################################################################
  
  import numpy as np
  
  yRow = yRow.flatten()
  yCol = yCol.flatten()
  yVal = yVal.flatten()
  
  # insert twin of each current entry without checking for its presence or
  # absence.
  yRow_symm = np.append(yRow,yCol)
  yCol_symm = np.append(yCol,yRow)
  yVal_symm = np.append(yVal,yVal)
  
  # only keep unique entries.
  entry = np.vstack((yRow_symm,yCol_symm))
  _,unique_entry_index = np.unique(entry,return_index=True,axis=1)
  yRow_symm = yRow_symm[unique_entry_index].reshape((1,-1))
  yCol_symm = yCol_symm[unique_entry_index].reshape((1,-1))
  yVal_symm = yVal_symm[unique_entry_index].reshape((1,-1))
  
  return yRow_symm,yCol_symm,yVal_symm

