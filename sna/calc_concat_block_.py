################################################################################
def calc_concat_block(block_type,n,id0,id1,c):
# 
# copyright (c) Russell Fung 2020
################################################################################
  
  from .block_file_template_ import block_file_template
  from .calc_block_          import calc_block
  from .read_block_          import read_block
  from .shift_and_add_       import shift_and_add
  from .write_block_         import write_block
  
  import os
  import numpy as np
  
  block_file = block_file_template(block_type).format(n,id0,id1,c)
  if os.path.exists(block_file):
    return
  
  if (block_type=="square"):
    row = id0
    col = id1
    diag = col-row
    block = row
    triu_pipe_block = read_block('pipe',n,diag,block,c)
    if (diag==0):
      tril_pipe_block = np.zeros((n,n))
    else:
      tril_pipe_block = read_block('pipe',n,diag-1,block,c)
    sandbox = np.hstack((tril_pipe_block,triu_pipe_block))
    
    measure = np.zeros((n,n))
    for jj in range(n):
      measure[jj,:] = sandbox[jj,n-jj:2*n-jj]
    
    if (diag==0):
      for jj in range(n):
        measure[jj,:jj] = measure[:jj,jj]
    
    write_block('square',n,row,col,c,measure)
  elif (block_type=="pipe"):
    to_store = bin(c)[2:][::-1]
    num_doubling = len(to_store)-1
    
    # stored_order is the current accumulated order
    stored_order = 0
    diag = id0
    block = id1
    calc_block('pipe',n,diag,block)
    
    for iter in range(num_doubling):
      # current_order is pre-doubling order
      current_order = np.power(2,iter)
      increment = current_order*int(to_store[iter])
      shift_and_add(n,diag,block,stored_order,increment)
      stored_order += increment
      shift_and_add(n,diag,block,current_order,current_order)
    increment = current_order*2
    shift_and_add(n,diag,block,stored_order,increment)
    stored_order += increment
  else:
    print(SnA_error('UnknownBlockType'))
    return

