################################################################################
def read_diag(n,diag,c):
# 
# copyright (c) Russell Fung 2023
################################################################################
  
  from misc_tools import read_h5
  from sna import diag_file_template
  import os
  
  diag_file = diag_file_template().format(n,diag,c)
  if not os.path.exists(diag_file):
    if (c==1):
      calc_diag(n,diag)
    else:
      calc_concat_diag(n,diag,c)
  measure = read_h5(diag_file,'measure')
  
  return measure

