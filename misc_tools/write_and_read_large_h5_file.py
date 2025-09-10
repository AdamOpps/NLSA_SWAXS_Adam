################################################################################
def write_and_read_large_h5_file(num_float):
# 
# copyright (c) Russell Fung 2024
################################################################################
  
  from misc_tools import read_h5,report_runtime,write_h5
  import numpy as np
  import os
  import subprocess
  import time
  
  t0 = time.time()
  job_id = '<create array>'
  A = np.random.uniform(size=num_float)
  t1 = time.time()
  report_runtime(job_id,t0,t1)
  
  data_file = 'random_number.h5'
  variable_name = 'random_vector'
  
  try:
    os.remove(data_file)
  except:
    pass
  
  try:
    t0 = time.time()
    job_id = '<write array>'
    write_h5(data_file,A,variable_name)
    t1 = time.time()
    report_runtime(job_id,t0,t1)
    print('#####')
    print('I can write {} floats to an H5 file'.format(num_float))
    result = subprocess.run(['ls','-lh',data_file],stdout=subprocess.PIPE)
    result = result.stdout.decode('utf-8')[:-1]
    print(result)
    try:
      t0 = time.time()
      job_id = '<clear array>'
      del A
      t1 = time.time()
      report_runtime(job_id,t0,t1)
      print('variable cleared')
    except:
      print('variable NOT cleared')
    try:
      t0 = time.time()
      job_id = '<read array>'
      B = read_h5(data_file,variable_name)
      t1 = time.time()
      report_runtime(job_id,t0,t1)
      print('#####')
      print('and I can read them back.')
    except:
      print('#####')
      print('but I cannot read them back.')
  except:
    print('#####')
    print('I cannot write {} floats to an H5 file'.format(num_float))
 
import os
import sys

cxfel_root = os.environ['CXFEL_ROOT']
startup_file = cxfel_root+'/misc_tools/startup.py'
exec(open(startup_file).read())

if __name__=="__main__":
  num_float = int(sys.argv[1])
  write_and_read_large_h5_file(num_float)

