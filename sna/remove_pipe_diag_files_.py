################################################################################
def remove_pipe_diag_files(n,diag,c,condition):
# 
# copyright (c) Russell Fung 2020
################################################################################
  
  if not condition:
    return
  
  from .block_file_template_ import block_file_template
  
  import glob
  import os
  
  block_file_list = glob.glob(block_file_template('pipe').format(n,diag,'*',c))
  for block_file in block_file_list:
    os.remove(block_file)

