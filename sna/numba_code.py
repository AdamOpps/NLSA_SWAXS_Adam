from numba import cuda

@cuda.jit('void(float64[:,:],float64[:],int32,int32)')
def cuda_add_column_to_matrix(A,B,nRow_A,nCol_A):
  row,col = cuda.grid(2)
  if (row>=nRow_A) or (col>=nCol_A):return
  A[row,col] += B[row]

@cuda.jit('void(float64[:,:],float64[:],int32,int32)')
def cuda_add_row_to_matrix(A,B,nRow_A,nCol_A):
  row,col = cuda.grid(2)
  if (row>=nRow_A) or (col>=nCol_A):return
  A[row,col] += B[col]

@cuda.jit('void(float64[:,:],float64[:],int32,int32)')
def cuda_col_sum_of_squares(A,S,nRow_A,nCol_A):
  col = cuda.grid(1)
  if (col>=nCol_A):return
  S[col] = 0
  for row in range(nRow_A):
    S[col] += A[row,col]*A[row,col]

@cuda.jit('void(float64[:,:],float64[:,:],int32,int32)')
def cuda_copy_matrix(A,C,nRow_A,nCol_A):
  row,col = cuda.grid(2)
  if (row>=nRow_A) or (col>=nCol_A):return
  C[row,col] = A[row,col]

@cuda.jit('void(float64[:,:],float64[:,:],float64[:,:],int32,int32,int32)')
def cuda_matmul(A,B,C,nRow_A,nCol_B,D):
  row,col = cuda.grid(2)
  if (row>=nRow_A) or (col>=nCol_B):return
  C[row,col] = 0
  for d in range(D):
    C[row,col] += A[row,d]*B[d,col]

@cuda.jit('void(float64[:,:],float64,int32,int32)')
def cuda_multiply_scalar_to_matrix(A,alpha,nRow_A,nCol_A):
  row,col = cuda.grid(2)
  if (row>=nRow_A) or (col>=nCol_A):return
  A[row,col] *= alpha

@cuda.jit('void(float64[:,:],float64[:],int32,int32)')
def cuda_row_sum_of_squares(A,S,nRow_A,nCol_A):
  row = cuda.grid(1)
  if (row>=nRow_A):return
  S[row] = 0
  for col in range(nCol_A):
    S[row] += A[row,col]*A[row,col]

@cuda.jit('void(float64[:,:],float64[:,:],int32,int32)')
def cuda_shift_and_add_double_c(A,S,n_curr,log2_c_curr):
  row,col = cuda.grid(2)
  shift = 2**log2_c_curr
  if (row>=n_curr-shift) or (col>=n_curr-shift):return
  S[row,col] = A[row,col]+A[row+shift,col+shift]

def numba_cdist(XA,XB,metric=None):
  import numpy as np
  from timeit import default_timer as timer
  
  runtime = np.empty(shape=[0,1])
  runtime = np.append(runtime,timer())
  
  nA,D = XA.shape
  nB = XB.shape[0]
  device = cuda.get_current_device()
  tpb = device.WARP_SIZE
  bpgA = (nA+tpb-1)//tpb
  bpgB = (nB+tpb-1)//tpb
  
  dev_A = cuda.to_device(XA)
  dev_B = cuda.to_device(XB.T)
  
  S1 = np.empty((nA)).astype(np.float64)
  dev_S1 = cuda.to_device(S1,copy=False)
  grid_dim = bpgA
  block_dim = tpb
  result_cuda = cuda_row_sum_of_squares[grid_dim,block_dim](dev_A,dev_S1,nA,D)
  
  S2 = np.empty((nB)).astype(np.float64)
  dev_S2 = cuda.to_device(S2,copy=False)
  grid_dim = bpgB
  block_dim = tpb
  result_cuda = cuda_col_sum_of_squares[grid_dim,block_dim](dev_B,dev_S2,D,nB)
  
  X = np.empty((nA,nB),dtype=np.float64)
  dev_X = cuda.to_device(X,copy=False)
  grid_dim = (bpgA,bpgB)
  block_dim = (tpb,tpb)
  result_cuda = cuda_matmul[grid_dim,block_dim](dev_A,dev_B,dev_X,nA,nB,D)
  
  result_cuda = cuda_multiply_scalar_to_matrix[grid_dim,block_dim](dev_X,-2,nA,nB)
  result_cuda = cuda_add_column_to_matrix[grid_dim,block_dim](dev_X,dev_S1,nA,nB)
  result_cuda = cuda_add_row_to_matrix[grid_dim,block_dim](dev_X,dev_S2,nA,nB)
  
  runtime = np.append(runtime,timer())
  
  dev_X.copy_to_host(X)
  
  runtime = np.append(runtime,timer())
  runtime_total = runtime[2]-runtime[0]
  runtime_device_to_host = runtime[2]-runtime[1]
  
  return X,runtime_total,runtime_device_to_host
  
def numba_cdist_SnA(XA,metric=None):
  import numpy as np
  from timeit import default_timer as timer
  
  runtime = np.empty(shape=[0,1])
  runtime = np.append(runtime,timer())
  
  nA,D = XA.shape
  nB = nA
  device = cuda.get_current_device()
  tpb = device.WARP_SIZE
  bpgA = (nA+tpb-1)//tpb
  bpgB = bpgA
  
  dev_A = cuda.to_device(XA)
  dev_B = cuda.to_device(XA.T)
  
  S1 = np.empty((nA)).astype(np.float64)
  dev_S1 = cuda.to_device(S1,copy=False)
  grid_dim = bpgA
  block_dim = tpb
  result_cuda = cuda_row_sum_of_squares[grid_dim,block_dim](dev_A,dev_S1,nA,D)
  
  S2 = np.empty((nB)).astype(np.float64)
  dev_S2 = cuda.to_device(S2,copy=False)
  grid_dim = bpgB
  block_dim = tpb
  result_cuda = cuda_col_sum_of_squares[grid_dim,block_dim](dev_B,dev_S2,D,nB)
  
  X = np.empty((nA,nB),dtype=np.float64)
  dev_X = cuda.to_device(X,copy=False)
  grid_dim = (bpgA,bpgB)
  block_dim = (tpb,tpb)
  result_cuda = cuda_matmul[grid_dim,block_dim](dev_A,dev_B,dev_X,nA,nB,D)
  
  result_cuda = cuda_multiply_scalar_to_matrix[grid_dim,block_dim](dev_X,-2,nA,nB)
  result_cuda = cuda_add_column_to_matrix[grid_dim,block_dim](dev_X,dev_S1,nA,nB)
  result_cuda = cuda_add_row_to_matrix[grid_dim,block_dim](dev_X,dev_S2,nA,nB)
  
  S = np.empty((nA,nA)).astype(np.float64)
  dev_S = cuda.to_device(S,copy=False)
  c_curr = 1
  log2_c_curr = 0
  n_curr = nA
  while c_curr<np.floor(nA/2):
    c_next = 2*c_curr
    log2_c_next = log2_c_curr+1
    n_next = n_curr-c_curr
    bpg_next = (n_next+tpb-1)//tpb
    grid_dim = (bpg_next,bpg_next)
    block_dim = (tpb,tpb)
    result_cuda = cuda_shift_and_add_double_c[grid_dim,block_dim](dev_X,dev_S,n_curr,log2_c_curr)
    result_cuda = cuda_copy_matrix[grid_dim,block_dim](dev_S,dev_X,n_next,n_next)
    c_curr,log2_c_curr,n_curr = c_next,log2_c_next,n_next
  
  runtime = np.append(runtime,timer())
  
  dev_S.copy_to_host(S)
  
  runtime = np.append(runtime,timer())
  runtime_total = runtime[2]-runtime[0]
  runtime_device_to_host = runtime[2]-runtime[1]
  
  return S[:n_curr,:n_curr],runtime_total,runtime_device_to_host
  
def numba_cdist_argsort(XA,XB,metric=None):
  import cupy
  import numpy as np
  from timeit import default_timer as timer
  
  runtime = np.empty(shape=[0,1])
  runtime = np.append(runtime,timer())
  nA,D = XA.shape
  nB = XB.shape[0]
  device = cuda.get_current_device()
  tpb = device.WARP_SIZE
  bpgA = (nA+tpb-1)//tpb
  bpgB = (nB+tpb-1)//tpb
  
  dev_A = cuda.to_device(XA)
  dev_B = cuda.to_device(XB.T)
  
  S1 = np.empty((nA)).astype(np.float64)
  dev_S1 = cuda.to_device(S1,copy=False)
  grid_dim = bpgA
  block_dim = tpb
  result_cuda = cuda_row_sum_of_squares[grid_dim,block_dim](dev_A,dev_S1,nA,D)
  
  S2 = np.empty((nB)).astype(np.float64)
  dev_S2 = cuda.to_device(S2,copy=False)
  grid_dim = bpgB
  block_dim = tpb
  result_cuda = cuda_col_sum_of_squares[grid_dim,block_dim](dev_B,dev_S2,D,nB)
  
  X = np.empty((nA,nB),dtype=np.float64)
  dev_X = cuda.to_device(X,copy=False)
  grid_dim = (bpgA,bpgB)
  block_dim = (tpb,tpb)
  result_cuda = cuda_matmul[grid_dim,block_dim](dev_A,dev_B,dev_X,nA,nB,D)
  
  C = np.zeros((nA,nB),dtype=np.float64)
  dev_C = cuda.to_device(C,copy=False)
  grid_dim = (bpgA,bpgB)
  block_dim = (tpb,tpb)
  result_cuda = cuda_multiply_scalar_to_matrix[grid_dim,block_dim](dev_X,-2,nA,nB)
  result_cuda = cuda_copy_matrix[grid_dim,block_dim](dev_X,dev_C,nA,nB)
  result_cuda = cuda_add_column_to_matrix[grid_dim,block_dim](dev_C,dev_S1,nA,nB)
  result_cuda = cuda_add_row_to_matrix[grid_dim,block_dim](dev_C,dev_S2,nA,nB)
  C_cupy = cupy.asarray(dev_C)
  O_cupy = cupy.argsort(C_cupy)
  runtime = np.append(runtime,timer())
  O = cupy.asnumpy(O_cupy)
  
  dev_C.copy_to_host(C)
  
  runtime = np.append(runtime,timer())
  runtime_total = runtime[2]-runtime[0]
  runtime_device_to_host = runtime[2]-runtime[1]
  
  return C,O,runtime_total,runtime_device_to_host
  
def numba_cdist_stub(nA,nB,D,gpu_runtime_total_max=20):
  import numpy as np
  from scipy.spatial.distance import cdist
  from timeit import default_timer as timer
  
  A = np.random.random((nA,D)).astype(np.float64)
  B = np.random.random((nB,D)).astype(np.float64)
  
  C_numba,gpu_runtime_total,gpu_runtime_device_to_host = numba_cdist(A,B,'sqeuclidean')
  print("Elapsed (cuda result on host)   = {:.4}s".format((gpu_runtime_total)))
  print("Elapsed (cuda result on device) = {:.4}s".format((gpu_runtime_total-gpu_runtime_device_to_host)))
  
  if (gpu_runtime_total>gpu_runtime_total_max): return None
  
  runtime = np.empty(shape=[0,1])
  runtime = np.append(runtime,timer())
  C_numpy = cdist(A,B,'sqeuclidean')
  runtime = np.append(runtime,timer())
  cpu_runtime = runtime[1]-runtime[0]
  print("Elapsed (numpy) = {:.4}s".format((cpu_runtime)))
  
  err = C_numba-C_numpy
  print("error = {:.4e}".format((np.max(np.abs(err)))))
  
  return gpu_runtime_total,gpu_runtime_device_to_host,cpu_runtime,err
 
def numba_cdist_SnA_stub(nA,D,gpu_runtime_total_max=20):
  import numpy as np
  from scipy.spatial.distance import cdist
  from timeit import default_timer as timer
  
  A = np.random.random((nA,D)).astype(np.float64)
  C_numba,gpu_runtime_total,gpu_runtime_device_to_host = numba_cdist_SnA(A,'sqeuclidean')
  print("Elapsed (cuda result on host)   = {:.4}s".format((gpu_runtime_total)))
  print("Elapsed (cuda result on device) = {:.4}s".format((gpu_runtime_total-gpu_runtime_device_to_host)))
  
  if (gpu_runtime_total>gpu_runtime_total_max): return None
  
  runtime = np.empty(shape=[0,1])
  runtime = np.append(runtime,timer())
  C_numpy = cdist(A,A,'sqeuclidean')
  
  runtime = np.append(runtime,timer())
  cpu_runtime = runtime[1]-runtime[0]
  print("Elapsed (numpy) = {:.4}s".format((cpu_runtime)))
  
  c_curr = 1
  log2_c_curr = 0
  n_curr = nA
  while c_curr<np.floor(nA/2):
    c_next = 2*c_curr
    log2_c_next = log2_c_curr+1
    n_next = n_curr-c_curr
    shift = 2**log2_c_curr
    C_numpy = C_numpy[:-shift,:-shift]+C_numpy[shift:,shift:]
    c_curr,log2_c_curr,n_curr = c_next,log2_c_next,n_next
  
  err = C_numba-C_numpy
  print("error = {:.4e}".format((np.max(np.abs(err)))))
  
  return gpu_runtime_total,gpu_runtime_device_to_host,cpu_runtime,err
  
def numba_cdist_argsort_stub(nA,nB,D,gpu_runtime_total_max=20):
  import numpy as np
  from scipy.spatial.distance import cdist
  from timeit import default_timer as timer
  
  A = np.random.random((nA,D)).astype(np.float64)
  B = np.random.random((nB,D)).astype(np.float64)
  
  C_numba,O_numba,gpu_runtime_total,gpu_runtime_device_to_host = numba_cdist_argsort(A,B,'sqeuclidean')
  print("Elapsed (cuda result on host)   = {:.4}s".format((gpu_runtime_total)))
  print("Elapsed (cuda result on device) = {:.4}s".format((gpu_runtime_total-gpu_runtime_device_to_host)))
  
  if (gpu_runtime_total>gpu_runtime_total_max): return None
  
  runtime = np.empty(shape=[0,1])
  runtime = np.append(runtime,timer())
  C_numpy = cdist(A,B,'sqeuclidean')
  O_numpy = np.argsort(C_numpy)
  runtime = np.append(runtime,timer())
  cpu_runtime = runtime[1] - runtime[0]
  print("Elapsed (numpy) = {:.4}s".format((cpu_runtime)))
  
  err_C = C_numba-C_numpy
  print("error = {:.4e}".format((np.max(np.abs(err_C)))))
  err_O = O_numba-O_numpy
  print("error = {}".format((np.max(np.abs(err_O)))))
  
  return gpu_runtime_total,gpu_runtime_device_to_host,cpu_runtime,err_C,err_O

def run_numba_cdist_stub():
  import matplotlib.pyplot as plt
  import numpy as np
  
  n_list                     = np.empty(shape=[0,1])
  D_list                     = np.empty(shape=[0,1])
  gpu_runtime_total          = np.empty(shape=[0,1])
  gpu_runtime_device_to_host = np.empty(shape=[0,1])
  cpu_runtime                = np.empty(shape=[0,1])
  err                        = np.empty(shape=[0,1])
  for n in np.power(2,range(2,17)):
    for D in np.power(2,range(2,17)):
      print("----------")
      print("n = {}".format((n)))
      print("D = {}".format((D)))
      result = numba_cdist_stub(nA=n,nB=n,D=D)
      if result is None: break
      n_list = np.append(n_list,n)
      D_list = np.append(D_list,D)
      gpu_runtime_total          = np.append(gpu_runtime_total,         result[0])
      gpu_runtime_device_to_host = np.append(gpu_runtime_device_to_host,result[1])
      cpu_runtime                = np.append(cpu_runtime,               result[2])
      err                        = np.append(err,                       result[3])
    of_interest = np.where(n_list==n)[0]
    x = D_list[of_interest]
    plt.figure()
    y = cpu_runtime[of_interest]
    plt.plot(x,y,'rx-',linewidth=2.0,fillstyle='none',label='cpu')
    y = gpu_runtime_total[of_interest]
    plt.plot(x,y,'bo-',linewidth=2.0,fillstyle='none',label='gpu host')
    y = gpu_runtime_total[of_interest]-gpu_runtime_device_to_host[of_interest]
    plt.plot(x,y,'gs-',linewidth=2.0,fillstyle='none',label='gpu device')
    plt.xscale('log',base=10)
    plt.yscale('log',base=10)
    plt.xlabel('#Pixels/Snapshot, D',fontsize=15)
    plt.ylabel('Squared Euclidean Distance Runtime (s)',fontsize=15)
    plt.title('#Snapshots, $N_A = N_B$ = {}'.format((n)),fontsize=20)
    plt.legend()
    plt.show(block=False)
    figure_name = 'runtime_n_{}.jpg'.format((n))
    plt.savefig(figure_name)
    plt.close()

