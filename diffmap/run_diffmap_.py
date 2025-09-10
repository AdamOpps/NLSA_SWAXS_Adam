################################################################################
def diffmap_analysis(yRow_yCol_yVal_file,sigma,nEigs,alpha):
# 
# copyright (c) Russell Fung 2019
# updated May 25, 2022
################################################################################
  
  import time
  from diffmap import analyze
  from misc_tools import read_h5,report_runtime
  import os
  
  t0 = time.time()
  job_id = '<Diffusion Map>'
  
  eigVec_eigVal_train_file = analyze(yRow_yCol_yVal_file,sigma,nEigs,alpha)
  eigVec = read_h5(eigVec_eigVal_train_file,'eigVec')
  eigVal = read_h5(eigVec_eigVal_train_file,'eigVal')
  column_sum = read_h5('column_sum_and_D.h5','column_sum')
  column_D = read_h5('column_sum_and_D.h5','column_D')
  # os.remove(yRow_yCol_yVal_file)
  os.remove(eigVec_eigVal_train_file)
  os.remove('column_sum_and_D.h5')
  
  t1 = time.time()
  report_runtime(job_id,t0,t1)
  
  return eigVec,eigVal,column_sum,column_D

import os
import sys

cxfel_root = os.environ['CXFEL_ROOT']
startup_file = cxfel_root+'/misc_tools/startup.py'
exec(open(startup_file).read())

if __name__=="__main__":
  yRow_yCol_yVal_file = sys.argv[1]
  sigma_factor        = sys.argv[2]
  nEigs               = sys.argv[3]
  alpha               = sys.argv[4]
  
  from misc_tools import read_h5,write_h5
  
  sigma_factor = float(sigma_factor)
  nEigs = int(nEigs)
  alpha = float(alpha)
  sigma_opt = read_h5('sigma_opt.h5','sigma_opt')
  sigma = sigma_factor*sigma_opt
  eigVec,eigVal,_,_ = diffmap_analysis(yRow_yCol_yVal_file,sigma,nEigs,alpha)
  
  write_h5('eigVec_eigVal.h5',eigVec,'eigVec')
  write_h5('eigVec_eigVal.h5',eigVal,'eigVal')
