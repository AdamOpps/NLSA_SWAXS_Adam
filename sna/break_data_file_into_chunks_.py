################################################################################
def break_data_file_into_chunks(n):
# 
# copyright (c) Russell Fung 2020
# updated May 11, 2022
################################################################################
  
  from .block_file_template_ import block_file_template
  from .read_run_info_       import read_run_info
  
  from misc_tools import read_h5,write_h5
  import ntpath
  import numpy as np
  
  data_file,variable_name,N,D,measure_type,_,h5,transpose = read_run_info()
  T = read_h5(data_file,variable_name,h5,transpose)[0:N,0:D]
  if ((measure_type=="dSqM")|(measure_type=="dotM")):
    M = read_h5(data_file,'mask',h5,transpose)[0:N,0:D]
  
  data_file_no_path = ntpath.basename(data_file)
  data_chunk_template = block_file_template("data_chunk").format(data_file_no_path.replace('.','_'),n,'{}')
  
  num_chunk,leftover = np.divmod(N,n)
  num_chunk += (leftover>0)
  
  for chunk in range(num_chunk):
    chunk_file = data_chunk_template.format(chunk)
    T_chunk = T[chunk*n:(chunk+1)*n,:]
    write_h5(chunk_file,T_chunk,'T_chunk')
    if ((measure_type=="dSqM")|(measure_type=="dotM")):
      M_chunk = M[chunk*n:(chunk+1)*n,:]
      write_h5(chunk_file,M_chunk,'M_chunk')

