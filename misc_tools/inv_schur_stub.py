def inv_schur_stub(p=10000,q=100):
  
  import numpy as np
  import time
  
  from inv_schur import inv_schur
  from report_runtime_ import report_runtime
  
  M = np.random.uniform(size=(p+q,p+q))
  A = M[:p,:p]
  B = M[:p,p:]
  C = M[p:,:p]
  D = M[p:,p:]
  t0 = time.time()
  R0 = np.linalg.inv(M)
  t1 = time.time()
  report_runtime('invert full matrix',t0,t1)
  t0 = time.time()
  R1 = inv_schur(A,B,C,D)
  t1 = time.time()
  report_runtime('invert full matrix, using Schur complement',t0,t1)
  err = np.max(np.abs(R0-R1).reshape([-1,1]))
  print(err)
  A_inv = np.linalg.inv(A) # inverse of the sub-matrix is not timed.
  t0 = time.time()
  R1 = inv_schur(A_inv,B,C,D,A_inv_provided=True)
  t1 = time.time()
  report_runtime('re-use inverse of sub-matrix, using Schur complement',t0,t1)
  err = np.max(np.abs(R0-R1).reshape([-1,1]))
  print(err)

if __name__=="__main__":
  inv_schur_stub()
