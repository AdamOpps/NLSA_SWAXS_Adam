def reconstruct(D,kOfInterest,tOfInterest,copyOfInterest,U,S,V):
    import numpy as np
    from math import floor
    num_k = len(kOfInterest)
    num_t = len(tOfInterest)
    num_cc = len(copyOfInterest)
    X = np.zeros((num_k,D,num_t))
    for ik in range(num_k):
        Xk = np.zeros((D,num_t))
        print('Reconstructing Mode {}'.format(ik))
        for cc in copyOfInterest:
            try:
                Xk = Xk + np.outer(np.outer(U[cc*D:(cc+1)*D,ik],S[ik,ik]),V[cc:cc+num_t,ik])
            except Exception as e:
                print(e)
        X[ik,:,:] = Xk/num_cc
    return X


def nlsa(data_file,variable_name,mu_psi_file,ell,N,D,n,c,num_copy,usv_file):
    import time
    from misc_tools import read_h5,write_h5
    from sna import read_block
    from math import ceil
    
    t0 = time.time()
    X1 = read_h5(data_file,variable_name).T
    mu = read_h5(mu_psi_file,'mu')
    psi = read_h5(mu_psi_file,'psi')
    nS = N-c+1
    psi_1 = np.concatenate((np.ones((1,nS)),psi))
    psi = psi_1
    del psi_1
    mu_psi = mu*psi
    mu_psi = mu_psi.T
    ATA = np.zeros((ell+1,ell+1))
    numRow = ceil(nS/n)
    numCol = numRow
    numFiles = numRow*numCol
    fileNum = 0
    for row in range(numRow):
        j0 = row*n
        j1 = j0+n
        j1 = min(j1,nS)
        XcTXc_mu_psi = np.zeros((n,ell+1))
        for col in range(numCol):
            i0 = col*n
            i1 = i0+n
            i1 = min(i1,nS)
            
            if (row>col):
                XcTXc = read_block('square',n,col,row,c)
                XcTXc = XcTXc.T
            else:
                XcTXc = read_block('square',n,row,col,c)
            
            if (col==numCol-1):
                n1 = nS%n
                XcTXc = XcTXc[:,:n1]
            
            XcTXc_mu_psi = XcTXc_mu_psi+np.matmul(XcTXc,mu_psi[i0:i1,:])
            fileNum = fileNum+1
            msg = 'processing files {0:.1f}% ...'.format(fileNum/numFiles*100)
            if (fileNum%np.floor(0.1*numFiles)==0):
                print(msg)
            
        if (row==numRow-1):
            n1 = nS%n
            XcTXc_mu_psi = XcTXc_mu_psi[:n1,:]

        ATA = ATA+np.matmul(mu_psi[j0:j1,:].T,XcTXc_mu_psi)

    [S_sq,EV]=eig((ATA+ATA.T)/2.)
    order = np.argsort(S_sq)[::-1]
    S_sq = S_sq[order]
    V = EV[:,order]

    S = np.sqrt(S_sq)
    invS = np.real(np.diag(1./S))
    S = np.real(np.diag(S))
    U = np.zeros((D * num_copy, ell + 1))
    for cc in range(1, num_copy + 1):  # MATLAB is 1-indexed
        row_idx = slice((cc - 1) * D, cc * D)
        col_indices = np.arange(c - 1, X1.shape[1]) - ((c - num_copy) / 2) - cc + 1
        col_indices = col_indices.astype(int)  # Ensure indices are integers
        U[row_idx, :] = X1[:, col_indices] @ mu_psi @ V @ invS


    V = np.matmul(psi.T,V) # V has a shape of nS*(ell+1)
    
    print("U,S and V are stored in file:"+usv_file)
    write_h5(usv_file,ATA,'ata')
    write_h5(usv_file,V,'V')
    write_h5(usv_file,S,'S')
    write_h5(usv_file,U,'U')
    t1 = time.time()
    print('Elapsed time {0:.2f} second'.format(t1-t0))
    try:
        X = reconstruct(D,np.arange(ell+1),np.arange(N-c+1-num_copy+1),np.arange(num_copy),U,S,V)
        write_h5(usv_file,X,'X_recon')
    except exception as e:
        print(e)

if __name__ == "__main__":
    # Initialization
    import os
    import numpy as np
    from scipy.linalg import eig
    import sys 
    import time

    cxfel_root = os.environ['CXFEL_ROOT']
    startup_file = cxfel_root+'/misc_tools/startup.py'
    exec(open(startup_file).read())
    data_file = sys.argv[1]
    variable_name = sys.argv[2]
    mu_psi_file = sys.argv[3]
    ell = sys.argv[4]
    N = sys.argv[5]
    D = sys.argv[6]
    n = sys.argv[7]
    c = sys.argv[8]
    num_copy = int(sys.argv[9])
    usv_file = sys.argv[10]

    ell = int(ell)
    N = int(N)
    D = int(D)
    n = int(n)
    c = int(c)
    nlsa(data_file,variable_name,mu_psi_file,ell,N,D,n,c,num_copy,usv_file)
