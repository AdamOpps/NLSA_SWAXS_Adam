################################################################################
def calc_block(block_type,n,id0,id1):
# 
# copyright (c) Russell Fung 2020
# updated May 11, 2022
################################################################################
  
  from .block_file_template_ import block_file_template
  from .read_block_          import read_block
  from .read_run_info_       import read_run_info
  from .write_block_         import write_block
  
  from misc_tools import read_h5,write_h5
  import os
  
  block_file = block_file_template(block_type).format(n,id0,id1,1)
  if os.path.exists(block_file):
    return
  
  import numpy as np
  
  if (block_type=="square"):
    import ntpath
    
    data_file,variable_name,N,D,measure_type,_,h5,transpose = read_run_info()
    row = id0
    col = id1
    
    data_file_no_path = ntpath.basename(data_file)
    data_chunk_template = block_file_template("data_chunk").format(data_file_no_path.replace('.','_'),n,'{}')
    row_chunk_file = data_chunk_template.format(row)
    col_chunk_file = data_chunk_template.format(col)
    if not os.path.exists(row_chunk_file):
      T = read_h5(data_file,variable_name,h5,transpose)[0:N,0:D]
      T_row = T[row*n:(row+1)*n,:]
      write_h5(row_chunk_file,T_row,'T_chunk')
      if ((measure_type=="dSqM")|(measure_type=="dotM")):
        M = read_h5(data_file,'mask',h5,transpose)[0:N,0:D]
        M_row = M[row*n:(row+1)*n,:]
        write_h5(row_chunk_file,M_row,'M_chunk')
    if not os.path.exists(col_chunk_file):
      T = read_h5(data_file,variable_name,h5,transpose)[0:N,0:D]
      T_col = T[col*n:(col+1)*n,:]
      write_h5(col_chunk_file,T_col,'T_chunk')
      if ((measure_type=="dSqM")|(measure_type=="dotM")):
        M = read_h5(data_file,'mask',h5,transpose)[0:N,0:D]
        M_col = M[col*n:(col+1)*n,:]
        write_h5(col_chunk_file,M_col,'M_chunk')
    T_row = read_h5(row_chunk_file,'T_chunk')
    T_col = read_h5(col_chunk_file,'T_chunk')
    
    if (measure_type=="dSq"):
      from scipy.spatial.distance import cdist
      
      measure = cdist(T_row,T_col,'sqeuclidean')
    elif (measure_type=="dot"):
      measure = np.dot(T_row,T_col.T)
    elif (measure_type=="dSqM"):
      M_row = read_h5(row_chunk_file,'M_chunk')
      M_col = read_h5(col_chunk_file,'M_chunk').T
      T_row = T_row  *M_row
      T_col = T_col.T*M_col
      term1 = np.matmul(np.power(T_row,2),M_col)
      term2 = -2*np.matmul(T_row,T_col)
      term3 = np.matmul(M_row,np.power(T_col,2))
      term4 = np.matmul(M_row,M_col)
      measure = term1+term2+term3
      measure[measure<0] = 0
      measure = measure/term4*D
    elif (measure_type=="dotM"):
      M_row = read_h5(row_chunk_file,'M_chunk')
      M_col = read_h5(col_chunk_file,'M_chunk').T
      T_row = T_row  *M_row
      T_col = T_col.T*M_col
      term1 = np.matmul(T_row,T_col)
      term2 = np.matmul(M_row,M_col)
      measure = term1/term2*D
    else:
      print(SnA_error('UnknownMeasureType'))
      return
    n_row,n_col = measure.shape
    if (n_row<n):
      measure = np.pad(measure,((0,n-n_row),(0,0)),'constant')
    if (n_col<n):
      measure = np.pad(measure,((0,0),(0,n-n_col)),'constant')
  elif (block_type=="pipe"):
    diag = id0
    block = id1
    
    row = block
    col_triu = diag+block
    col_tril = col_triu+1
    
    triu_square_block = read_block('square',n,row,col_triu,1)
    tril_square_block = read_block('square',n,row,col_tril,1)
    
    measure = np.zeros((n,n))
    sandbox = np.hstack((triu_square_block,tril_square_block))
    for jj in range(n):
      measure[jj,:] = sandbox[jj,jj:jj+n]
  else:
    print(SnA_error('UnknownBlockType'))
    return
  
  write_block(block_type,n,id0,id1,1,measure)

