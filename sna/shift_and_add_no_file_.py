################################################################################
def shift_and_add_no_file(N,n,diag,a,b,measure_a,measure_b):
# 
# copyright (c) Russell Fung 2023
################################################################################
  
  import numpy as np
  import time
  
  runtime_read  = 0.0
  runtime_add   = 0.0
  runtime_write = 0.0
  
  if (b==0):
    # no increment,return stored_order as-is
    t0 = time.time()
    measure_c = measure_a
    t1 = time.time()
    runtime_read = t1-t0
  elif (a==0):
    # nothing stored so far
    # increment nonzero,return increment as new stored_order
    t0 = time.time()
    measure_c = measure_b
    t1 = time.time()
    runtime_read = t1-t0
  else:
    num_super = N-a-b+1
    t0 = time.time()
    measure_a = measure_a[:num_super-diag*n,:]
    measure_b = measure_b[a:a+num_super-diag*n,:]
    t1 = time.time()
    runtime_read = t1-t0
    
    t0 = time.time()
    measure_c = measure_a+measure_b
    t1 = time.time()
    runtime_add = t1-t0
  
  return measure_c,np.array([runtime_read,runtime_add,runtime_write])

