################################################################################
def shift_and_add(n,diag,block,a,b):
# 
# copyright (c) Russell Fung 2020
################################################################################
  
  import numpy as np
  
  runtime_read  = 0.0
  runtime_add   = 0.0
  runtime_write = 0.0
  
  if (a*b==0):
    return np.array([runtime_read,runtime_add,runtime_write])
  
  import os
  import time
  
  if (block is None):
    from sna import diag_file_template
    
    diag_file = diag_file_template().format(n,diag,a+b)
    if os.path.exists(diag_file): return np.array([runtime_read,runtime_add,runtime_write])
    
    from sna import read_run_info,read_diag,write_diag
    
    _,_,N,_,_,_,_,_ = read_run_info()
    
    t0 = time.time()
    num_super = N-a-b+1
    measure_a = read_diag(n,diag,a)[:num_super-diag*n,:]
    measure_b = read_diag(n,diag,b)[a:a+num_super-diag*n,:]
    t1 = time.time()
    runtime_read = t1-t0
    
    t0 = time.time()
    measure_c = measure_a+measure_b
    t1 = time.time()
    runtime_add = t1-t0
    
    t0 = time.time()
    write_diag(n,diag,a+b,measure_c)
    t1 = time.time()
    runtime_write = t1-t0
    
    return np.array([runtime_read,runtime_add,runtime_write])
  
  from sna import block_file_template
  
  block_file = block_file_template('pipe').format(n,diag,block,a+b)
  if os.path.exists(block_file): return np.array([runtime_read,runtime_add,runtime_write])
  
  from sna import read_block,write_block
  
  t0 = time.time()
  measure = read_block('pipe',n,diag,block,a)
  block_offset,shift = np.divmod(a,n)
  sandbox = read_block('pipe',n,diag,block+block_offset,b)[shift:,:]
  if (shift>0):
    leaked = read_block('pipe',n,diag,block+block_offset+1,b)[:shift,:]
    sandbox = np.vstack((sandbox,leaked))
  t1 = time.time()
  runtime_read = t1-t0
  
  t0 = time.time()
  measure += sandbox
  t1 = time.time()
  runtime_add = t1-t0
  
  t0 = time.time()
  write_block('pipe',n,diag,block,a+b,measure)
  t1 = time.time()
  runtime_write = t1-t0
  
  return np.array([runtime_read,runtime_add,runtime_write])
