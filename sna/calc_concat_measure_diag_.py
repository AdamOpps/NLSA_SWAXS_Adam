################################################################################
def calc_concat_measure_diag(n,diag,no_block=False):
# 
# copyright (c) Russell Fung 2020
################################################################################
  import numpy as np
  
  from sna import calc_block,read_run_info
  _,_,N,_,_,c,_,_ = read_run_info()
  
  num_block,leftover = np.divmod(N,n)
  num_block += (leftover>0)
  
  if (c==1):
    for block in range(num_block-diag):
      row = block
      col = diag+row
      calc_block('square',n,row,col)
    
    return
  
  from misc_tools import report_runtime
  from sna import remove_block_file
  
  runtime = np.array([0.0,0.0,0.0])
  
  if (no_block):
    from sna import calc_diag,read_diag,shift_and_add_no_file
    job_id = '<Shift-and-Add, no blocks, {}>'
    calc_diag(n,diag)
    measure_current = read_diag(n,diag,1)
  else:
    from sna import remove_pipe_diag_files,shift_and_add
    job_id = '<Shift-and-Add, with blocks, {}>'
    for block in range(num_block-diag): calc_block('pipe',n,diag,block)
  
  for block in range(num_block-diag):
    remove_block_file('square',n,block,diag+block,1,True)
    remove_block_file('square',n,block,diag+block+1,1,True)
  
  to_store = bin(c)[2:][::-1]
  num_doubling = len(to_store)-1
  
  # stored_order is the current accumulated order
  stored_order = 0
  if (no_block): measure_stored = np.zeros((N,n))
  
  for iter in range(num_doubling):
    # current_order is pre-doubling order
    current_order = np.power(2,iter)
    increment = current_order*int(to_store[iter])
    if (no_block):
      measure_stored,my_runtime = shift_and_add_no_file(N,n,diag,\
                                    stored_order,increment,\
                                    measure_stored,measure_current)
      runtime+=my_runtime
    else:
      num_block,leftover = np.divmod(N-stored_order-increment+1,n)
      num_block += (leftover>0)
      for block in range(num_block-diag):
        runtime+=shift_and_add(n,diag,block,stored_order,increment)
      remove_pipe_diag_files(n,diag,stored_order,(increment>0)&(stored_order>0))
    stored_order += increment
    if (no_block):
      measure_current,my_runtime = shift_and_add_no_file(N,n,diag,\
                                    current_order,current_order,\
                                    measure_current,measure_current)
      runtime+=my_runtime
    else:
      num_block,leftover = np.divmod(N-current_order-current_order+1,n)
      num_block += (leftover>0)
      for block in range(num_block-diag):
        runtime+=shift_and_add(n,diag,block,current_order,current_order)
      remove_pipe_diag_files(n,diag,current_order,not (stored_order==current_order))
  increment = current_order*2
  
  num_super = N-c+1
  num_block,leftover = np.divmod(num_super,n)
  num_block += (leftover>0)
  num_block -= diag
  
  if (no_block):
    from sna import write_block
    
    measure_stored,my_runtime = shift_and_add_no_file(N,n,diag,\
                                  stored_order,increment,\
                                  measure_stored,measure_current)
    runtime+=my_runtime
    
    measure_no_block = measure_stored
    for block in range(num_block):
      measure = measure_no_block[block*n:(block+1)*n,:]
      measure = np.pad(measure,((0,n-measure.shape[0]),(0,0)),'constant')
      write_block('pipe',n,diag,block,c,measure)
  else:
    for block in range(num_block):
      runtime+=shift_and_add(n,diag,block,stored_order,increment)
    remove_pipe_diag_files(n,diag,increment,stored_order>0)
    remove_pipe_diag_files(n,diag,stored_order,stored_order>0)
  stored_order += increment
  
  try:
    # return without reporting runtime if DO_NOT_REPORT_SNA_RUNTIME_DETAIL is defined
    import os
    do_not_report = os.environ['DO_NOT_REPORT_SNA_RUNTIME_DETAIL']
    return
  except:
    pass
  
  report_runtime(job_id.format('Reading'),0,runtime[0])
  report_runtime(job_id.format('Adding'),0,runtime[1])
  report_runtime(job_id.format('Writing'),0,runtime[2])

