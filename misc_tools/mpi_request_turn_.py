################################################################################
def mpi_request_turn(comm,taskID):
# 
# copyright (c) Russell Fung 2023
################################################################################
  
  if (comm is None): return
  
  my_rank = comm.Get_rank()
  
  if (my_rank>0):
    # worker requests a turn to perform task
    comm.send(my_rank,dest=0,tag=taskID+1)
    # worker waits for signal from root to proceed
    comm.recv(source=0,tag=taskID+2)

