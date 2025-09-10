def inv_schur(A,B,C,D,A_inv_provided=False):
  
  import numpy as np
  
  if A_inv_provided:
   A_inv = A
  else:
    A_inv = np.linalg.inv(A)
  schur_complement_A = D-C@A_inv@B
  A_inv_B = A_inv@B
  C_A_inv = C@A_inv
  R22 = np.linalg.inv(schur_complement_A)
  R12 = -A_inv_B@R22
  R21 = -R22@C_A_inv
  R11 = A_inv-R12@C_A_inv
  
  R = np.block([[R11,R12],[R21,R22]])
  return R
