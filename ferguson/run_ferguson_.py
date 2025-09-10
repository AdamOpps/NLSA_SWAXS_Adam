################################################################################
def ferguson_analysis(yRow_yCol_yVal_file):
# 
# copyright (c) Russell Fung 2019
# updated May 24, 2022
################################################################################
  
  import time
  import numpy as np
  
  from ferguson import analyze
  from misc_tools import report_runtime
  
  t0 = time.time()
  job_id = '<Ferguson Analysis>'
  
  sigma_opt,dimensionality = analyze(yRow_yCol_yVal_file)
  sigma_opt = 1.0e-4*np.round(1.0e4*sigma_opt)
  
  t1 = time.time()
  report_runtime(job_id,t0,t1)
  
  return sigma_opt

import os
import sys

cxfel_root = os.environ['CXFEL_ROOT']
startup_file = cxfel_root+'/misc_tools/startup.py'
exec(open(startup_file).read())

if __name__=="__main__":
  yRow_yCol_yVal_file = sys.argv[1]
  
  sigma_opt = ferguson_analysis(yRow_yCol_yVal_file)
  
  from misc_tools import write_h5
  write_h5('sigma_opt.h5',sigma_opt,'sigma_opt')

