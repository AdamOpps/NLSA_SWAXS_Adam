################################################################################
def run_sna(data_file,variable_name,N,D,measure_type='dSq',
  c=1,h5=True,transpose=False,n=500,nN=1000,sqDist_file='sqDist.h5',
  cleanup=False,no_block=True,run_mpi=True):
# 
# copyright (c) Russell Fung 2020
################################################################################
  
  if run_mpi:
    from mpi4py import MPI
    
    comm = MPI.COMM_WORLD
    my_rank = comm.Get_rank()
    group_size = comm.Get_size()
  else:
    comm = None
    my_rank = 0
    group_size = 1
  
  from misc_tools import mpi_release_turn,mpi_request_turn,mpi_roll_call,\
                         report_runtime
  import ntpath
  import numpy as np
  import os
  import shutil
  from sna import block_file_template,break_data_file_into_chunks,\
                  calc_concat_measure_diag,write_run_info
  import subprocess
  import time
  
  link_data_file_chunk_files = 1000
  collect_block_files        = 1010
  worker_directory_removed   = 1020
  square_files_available     = 1030
  
  cwd = os.getcwd()+'/'
  worker_directory = cwd+'sna_worker_{}/'.format(my_rank)
  if os.path.isdir(worker_directory):
    shutil.rmtree(worker_directory)
  os.mkdir(worker_directory)
  
  process = subprocess.Popen(['hostname'],stdout=subprocess.PIPE)
  my_hostname,err = process.communicate()
  my_hostname = my_hostname.decode('utf-8')
  worker_id = 'worker %d'%my_rank+' of %d'%group_size+' (%s'%my_hostname[:-1]+')'
  if run_mpi: worker_id = 'MPI-'+worker_id
  
  os.chdir(worker_directory)
  write_run_info(data_file,variable_name,N,D,measure_type,c,h5,transpose)
  
  num_super = N-c+1
  num_block,leftover = np.divmod(num_super,n)
  num_block += (leftover>0)
  
  data_file_no_path = ntpath.basename(data_file)
  data_chunk_template = block_file_template("data_chunk").format(data_file_no_path.replace('.','_'),n,'{}')
  
  num_chunk,leftover = np.divmod(N,n)
  num_chunk += (leftover>0)
  
  if (my_rank==0):
    os.chdir(cwd)
    write_run_info(data_file,variable_name,N,D,measure_type,c,h5,transpose)
    break_data_file_into_chunks(n)
    os.chdir(worker_directory)
  
  mpi_request_turn(comm,taskID=link_data_file_chunk_files)
  
  os.link(cwd+data_file,worker_directory+data_file)
  for chunk in range(num_chunk):
    chunk_file = data_chunk_template.format(chunk)
    os.link(cwd+chunk_file,worker_directory+chunk_file)
  
  mpi_release_turn(comm,taskID=link_data_file_chunk_files)
  
  t0 = time.time()
  job_id = '<Shift-and-Add>'
  
  my_diag = range(my_rank,num_block,group_size)
  for diag in my_diag:
    calc_concat_measure_diag(n,diag,no_block=no_block)
  
  os.chdir(cwd)
  
  t1 = time.time()
  report_runtime(worker_id+': '+job_id,t0,t1)
  
  mpi_request_turn(comm,taskID=collect_block_files)
  
  for diag in my_diag:
    for block in range(num_block-diag):
      if (c==1):
        row = block
        col = diag+row
        block = block_file_template("square").format(n,row,col,c)
      else:
        block = block_file_template("pipe").format(n,diag,block,c)
      shutil.move(worker_directory+block,cwd+block)
  
  mpi_release_turn(comm,taskID=collect_block_files)
  
  shutil.rmtree(worker_directory)
  
  mpi_roll_call(comm,taskID=worker_directory_removed)
  
  from sna import collect_concat_measure
  from sparsify import smallest_items_in_each_row_of_table
  
  if (c>1):
    from sna import pipe2square
    pipe2square(N,c,n,my_diag=my_diag)
  
  mpi_roll_call(comm,taskID=square_files_available)
  return 
  my_row = range(my_rank,num_block,group_size)
  
  if (my_rank==0):
    yRow = np.array([[]],dtype=int)
    yCol = np.array([[]],dtype=int)
    yVal = np.array([[]])
  
  # yRow,yCol 1-based
  for row in my_row:
    yVal_mine = collect_concat_measure(n,row=row)
    yRow_mine,yCol_mine,yVal_mine = \
      smallest_items_in_each_row_of_table(yVal_mine,nN)
    yRow_mine += row*n
    if (my_rank==0):
      yRow = np.hstack((yRow,yRow_mine))
      yCol = np.hstack((yCol,yCol_mine))
      yVal = np.hstack((yVal,yVal_mine))
      for worker in range(1,group_size):
        if (row+worker==num_block): break
        yRow_worker = comm.recv(source=worker,tag=0)
        yCol_worker = comm.recv(source=worker,tag=1)
        yVal_worker = comm.recv(source=worker,tag=2)
        comm.recv(source=worker,tag=3)
        yRow = np.hstack((yRow,yRow_worker))
        yCol = np.hstack((yCol,yCol_worker))
        yVal = np.hstack((yVal,yVal_worker))
    else:
      comm.send(yRow_mine,dest=0,tag=0)
      comm.send(yCol_mine,dest=0,tag=1)
      comm.send(yVal_mine,dest=0,tag=2)
      comm.send(  my_rank,dest=0,tag=3)
  
  if (my_rank==0):
    from misc_tools import write_h5
    from sparsify import symmetrize
    
    yRow,yCol,yVal = symmetrize(yRow,yCol,yVal)
    write_h5(sqDist_file,yRow,'yRow')
    write_h5(sqDist_file,yCol,'yCol')
    write_h5(sqDist_file,yVal,'yVal')
    
    if cleanup:
      from sna import post_sna_cleanup
      
      os.remove('.sna_run_info')
      post_sna_cleanup('data_chunk',data_file,n=n,c=c)
      if (c>1): post_sna_cleanup('pipe',data_file,n=n,c=c)
      post_sna_cleanup('square',data_file,n=n,c=c)

import os
import sys

cxfel_root = os.environ['CXFEL_ROOT']
startup_file = cxfel_root+'/misc_tools/startup.py'
exec(open(startup_file).read())

if __name__=="__main__":
  data_file     = sys.argv[1]
  variable_name = sys.argv[2]
  N             = sys.argv[3]
  D             = sys.argv[4]
  measure_type  = sys.argv[5]
  c             = sys.argv[6]
  h5            = sys.argv[7]
  transpose     = sys.argv[8]
  n             = sys.argv[9]
  nN            = sys.argv[10]
  sqDist_file   = sys.argv[11]
  cleanup       = sys.argv[12]
  no_block      = sys.argv[13]
  run_mpi       = sys.argv[14]
  
  N = int(N)
  D = int(D)
  c = int(c)
  h5 = h5=='True'
  transpose = transpose=='True'
  n = int(n)
  nN = int(nN)
  cleanup = cleanup=='True'
  no_block = no_block=='True'
  run_mpi = run_mpi=='True'
  run_sna(data_file,variable_name,N,D,measure_type,c,h5,transpose,n,nN,sqDist_file,cleanup,no_block,run_mpi)
