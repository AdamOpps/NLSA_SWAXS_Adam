################################################################################
def mpi_release_turn(comm,taskID):
# 
# copyright (c) Russell Fung 2023
################################################################################
  
  if (comm is None): return
  
  my_rank = comm.Get_rank()
  group_size = comm.Get_size()
  
  if (my_rank>0):
    # worker releases turn
    comm.send(my_rank,dest=0,tag=taskID+3)
    # worker waits for signal from root to move on
    comm.recv(source=0,tag=taskID+4)
  else:
    for worker in range(1,group_size):
      # root receives worker's request for a turn to perform task
      comm.recv(source=worker,tag=taskID+1)
      # root sends signal for worker to proceed
      comm.send(0,dest=worker,tag=taskID+2)
      # root waits for worker to release turn
      comm.recv(source=worker,tag=taskID+3)
      # root sends signal for worker to move on
      comm.send(0,dest=worker,tag=taskID+4)

