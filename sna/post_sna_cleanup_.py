################################################################################
def post_sna_cleanup(block_type,data_file,n,c):
# 
# copyright (c) Russell Fung 2022
################################################################################
  
  from sna import block_file_template
  
  import glob
  import os
  
  wildcard = block_file_template(block_type)
  
  if (block_type=="square"):
    fileList = glob.glob(wildcard.format(n,'*','*',c))
  elif (block_type=="pipe"):
    fileList = glob.glob(wildcard.format(n,'*','*',c))
  elif (block_type=="data_chunk"):
    fileList = glob.glob(wildcard.format(data_file.replace('.','_'),n,'*'))
  
  for file in fileList: 
    os.remove(file)

import os
import sys

cxfel_root = os.environ['CXFEL_ROOT']
startup_file = cxfel_root+'/misc_tools/startup.py'
exec(open(startup_file).read())

if __name__=="__main__":
  block_type = sys.argv[1]
  data_file  = sys.argv[2]
  n          = sys.argv[3]
  c          = sys.argv[4]
  
  n = int(n)
  c = int(c)
  post_sna_cleanup(block_type=block_type,data_file=data_file,n=n,c=c)

