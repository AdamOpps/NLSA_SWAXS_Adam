################################################################################
def remove_block_file(block_type,n,id0,id1,c,condition):
# 
# copyright (c) Russell Fung 2020
################################################################################
  
  if not condition:
    return
  
  from .block_file_template_ import block_file_template
  
  import os
  
  block_file = block_file_template(block_type).format(n,id0,id1,c)
  if os.path.exists(block_file):
    os.remove(block_file)

