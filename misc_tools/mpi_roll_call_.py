################################################################################
def mpi_roll_call(comm,taskID):
# 
# copyright (c) Russell Fung 2023
################################################################################
  
  if (comm is None): return
  
  my_rank = comm.Get_rank()
  group_size = comm.Get_size()
  
  if (my_rank>0):
    # worker checks in
    comm.send(my_rank,dest=0,tag=taskID+1)
    # worker waits for signal from root to move on
    comm.recv(source=0,tag=taskID+2)
  else:
    for worker in range(1,group_size):
      # root checks in worker
      comm.recv(source=worker,tag=taskID+1)
    # root checks in all workers, before releasing them
    for worker in range(1,group_size):
      # root releases worker
      comm.send(0,dest=worker,tag=taskID+2)

