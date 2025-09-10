################################################################################
def write_block(block_type,n,id0,id1,c,measure):
# 
# copyright (c) Russell Fung 2020
# updated April 27, 2022
################################################################################
  
  from .block_file_template_ import block_file_template
  
  from misc_tools import write_h5
  
  block_file = block_file_template(block_type).format(n,id0,id1,c)
  write_h5(block_file,measure,'measure')

