################################################################################
def read_run_info():
# 
# copyright (c) Russell Fung 2020
# updated May 11, 2022
################################################################################
  
  run_info = open(".sna_run_info","r")
  data_file         = run_info.readline()[0:-1]
  variable_name     = run_info.readline()[0:-1]
  N                 = run_info.readline()
  D                 = run_info.readline()
  measure_type      = run_info.readline()[0:-1]
  c                 = run_info.readline()
  h5                = run_info.readline()[0:-1]=='True'
  transpose         = run_info.readline()[0:-1]=='True'
  run_info.close()
  
  N = int(N)
  D = int(D)
  c = int(c)
  
  return data_file,variable_name,N,D,measure_type,c,h5,transpose

