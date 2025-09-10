################################################################################
def pipe2square(N,c,n,my_diag=None):
# 
# copyright (c) Russell Fung 2022
################################################################################
  
  from sna import calc_concat_block
  
  import numpy as np
  
  num_super = N-c+1
  num_block,leftover = np.divmod(num_super,n)
  num_block += (leftover>0)
  
  if (my_diag is None): my_diag = range(num_block)
  
  for diag in my_diag:
    for block in range(num_block-diag):
      row = block
      col = diag+row
      calc_concat_block('square',n,row,col,c)

import os
import sys

cxfel_root = os.environ['CXFEL_ROOT']
startup_file = cxfel_root+'/misc_tools/startup.py'
exec(open(startup_file).read())

if __name__=="__main__":
  N             = sys.argv[1]
  c             = sys.argv[2]
  n             = sys.argv[3]
  
  N = int(N)
  c = int(c)
  n = int(n)
  pipe2square(N,c,n)

