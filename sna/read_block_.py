################################################################################
def read_block(block_type,n,id0,id1,c):
# 
# copyright (c) Russell Fung 2020
# updated May 11, 2022
################################################################################
  
  from .block_file_template_ import block_file_template
  from .calc_block_          import calc_block
  from .calc_concat_block_   import calc_concat_block
  from .read_run_info_       import read_run_info
  
  from misc_tools import read_h5
  import numpy as np
  
  _,_,N,_,_,_,_,_ = read_run_info()
  num_super = N-c+1
  num_block,leftover = np.divmod(num_super,n)
  num_block += (leftover>0)
  
  measure = np.zeros((n,n))
  if (block_type=="square"):
    row = id0
    col = id1
    max_block = num_block-1
    if ((row>max_block)|(col>max_block)):
      return measure
  elif (block_type=="pipe"):
    diag = id0
    block = id1
    max_diag = num_block-1
    max_block = max_diag-diag
    if ((diag>max_diag)|(block>max_block)):
      return measure
  
  import os
  
  block_file = block_file_template(block_type).format(n,id0,id1,c)
  if not os.path.exists(block_file):
    if (c==1):
      calc_block(block_type,n,id0,id1)
    else:
      calc_concat_block(block_type,n,id0,id1,c)
  measure = read_h5(block_file,'measure')
  
  return measure

