################################################################################
def collect_concat_measure(n,row=None):
# 
# copyright (c) Russell Fung 2020
################################################################################
  
  from .block_file_template_ import block_file_template
  from .read_block_          import read_block
  from .read_run_info_       import read_run_info
  
  import numpy as np
  import os
  
  _,_,N,_,_,c,_,_ = read_run_info()
  num_super = N-c+1
  num_block,leftover = np.divmod(num_super,n)
  num_block += (leftover>0)
  n_edge_block = leftover+n*(leftover==0)
  
  if not (row is None):
    concat_measure = np.zeros((n,num_super))
    for col in range(num_block):
      if (col>row):
        measure = read_block('square',n,row,col,c)
      elif (col<row):
        measure = read_block('square',n,col,row,c).T
      else:
        if (c==1):
          measure = read_block('square',n,row,col,1)
        else:
          diag = 0
          block = row
          triu_pipe_block = read_block('pipe',n,diag,block,c)
          measure = np.zeros((n,n))
          for jj in range(n):
            measure[jj,jj:] = triu_pipe_block[jj,:n-jj]
            measure[jj,:jj] = measure[:jj,jj]
      if (col==num_block-1): measure = measure[:,:n_edge_block]
      concat_measure[:,col*n:(col+1)*n] = measure
    if (row==num_block-1):
      concat_measure = concat_measure[:n_edge_block,:]
    
    return concat_measure
  
  concat_measure = np.zeros((num_super,num_super))
  
  if (c==1):
    for row in range(num_block):
      measure = read_block('square',n,row,row,1)
      if (row==num_block-1):
        measure = measure[:n_edge_block,:n_edge_block]
      concat_measure[row*n:(row+1)*n,row*n:(row+1)*n] = measure
      for col in range(row+1,num_block):
        measure = read_block('square',n,row,col,1)
        if (row==num_block-1):
          measure = measure[:n_edge_block,:]
        if (col==num_block-1):
          measure = measure[:,:n_edge_block]
        concat_measure[row*n:(row+1)*n,col*n:(col+1)*n] = measure
        concat_measure[col*n:(col+1)*n,row*n:(row+1)*n] = measure.T
    
    return concat_measure
  
  for row in range(num_block):
    block_file = block_file_template('square').format(n,row,row,c)
    if os.path.exists(block_file):
      measure = read_block('square',n,row,row,c)
    else:
      diag = 0
      block = row
      triu_pipe_block = read_block('pipe',n,diag,block,c)
      measure = np.zeros((n,n))
      for jj in range(n):
        measure[jj,jj:] = triu_pipe_block[jj,:n-jj]
        measure[jj,:jj] = measure[:jj,jj]
    if (row==num_block-1):
      measure = measure[:n_edge_block,:n_edge_block]
    concat_measure[row*n:(row+1)*n,row*n:(row+1)*n] = measure
    
    for col in range(row+1,num_block):
      measure = read_block('square',n,row,col,c)
      if (row==num_block-1):
        measure = measure[:n_edge_block,:]
      if (col==num_block-1):
        measure = measure[:,:n_edge_block]
      concat_measure[row*n:(row+1)*n,col*n:(col+1)*n] = measure
      concat_measure[col*n:(col+1)*n,row*n:(row+1)*n] = measure.T
     
  return concat_measure

