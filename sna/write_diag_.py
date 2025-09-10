################################################################################
def write_diag(n,diag,c,measure):
# 
# copyright (c) Russell Fung 2023
################################################################################
  
  from misc_tools import write_h5
  from sna import diag_file_template
  
  diag_file = diag_file_template().format(n,diag,c)
  write_h5(diag_file,measure,'measure')

