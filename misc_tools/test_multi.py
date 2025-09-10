import os 
import sys
import numpy as np
cxfel_root = '/home/uwm/huang229/Data/huang229/CXFEL/'
sys.path.append(cxfel_root)

from misc_tools import read_h5
xyz = read_h5(cxfel_root+'/data/DM_Closed_Loop.mat','Y',h5=False,transpose=True)

N,D = xyz.shape
sqDist_file = 'dSq'
nN = [50,100,500]
sigma_factor = [2,10]
c = 1

from misc_tools import write_h5
data_file = 'data_file_for_sna.h5'
variable_name = 'NxD_matrix_for_sna'
write_h5(data_file,xyz,variable_name)

import subprocess 
num_worker = 4
n = 250
sq_code = cxfel_root+"misc_tools/multi_nN_prepare_dsq.py"
subprocess.run(["mpiexec","-N",str(num_worker),"python",sq_code,data_file,
                variable_name,str(N),str(D),'dSq',str(c),"True","False",
                str(n),str(nN),sqDist_file,"True","True","True"])

from ferguson import ferguson_analysis
sigma_opt_list = []
for i,nN_i in enumerate(nN):
    sqDist_file_name = sqDist_file + '_nN_' + str(nN_i) + '.h5'
    sigma_opt = ferguson_analysis(sqDist_file_name)
    sigma_opt_list.append(sigma_opt)
write_h5('sigma_opt.h5',sigma_opt_list,'sigma_opt')

from diffmap import diffmap_analysis
nEigs = 5
n_nN = len(nN)
n_sigma = len(sigma_factor)
eigVec_list = []
for i,nN_i in enumerate(nN):
    eigVec_sigma_list = []
    sqDist_file_name = sqDist_file + '_nN_' + str(nN_i) + '.h5'
    for j,sigma_j in enumerate(sigma_factor):
        sigma = sigma_opt_list[i]*sigma_j
        eigVec,eigVal,_,_ = diffmap_analysis(sqDist_file_name,sigma,nEigs,1.0)
        eigVec_sigma_list.append(eigVec)
    eigVec_list.append(eigVec_sigma_list)
write_h5('eigVec_list.h5',eigVec_list,'list')

def corr_no_noisy(a):
    num_eig = a.shape[1]
    non_noise_list = []
    for i in range(1,num_eig):
        if np.max(a[:10,i])-np.min(a[:10,i]) < 9e-4:
            non_noise_list.append(i)

    return non_noise_list

# from scipy.stats import pearsonr
eigVec_corr = np.zeros((n_nN*n_sigma,n_nN*n_sigma))
bar=0.9
for i_nN,_ in enumerate(nN):
    for j_nN,_ in enumerate(nN):
        for i_sigma,_ in enumerate(sigma_factor):
            ev1 = eigVec_list[i_nN][i_sigma]
            for j_sigma,_ in enumerate(sigma_factor):
                ev2 = eigVec_list[j_nN][j_sigma]
                non_noise = list(set(corr_no_noisy(ev1))&set(corr_no_noisy(ev2)))
                # corr = pearsonr(ev1,ev2)
                eigVec_corr[i_nN*n_sigma+i_sigma,j_nN*n_sigma+j_sigma] = np.count_nonzero(np.abs(np.dot(ev1.T,ev2))>bar)

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
fig, ax = plt.subplots()
im = ax.imshow(eigVec_corr,extent=[0,n_nN*n_sigma,n_nN*n_sigma,0])
ax.set_xlim(0,n_nN*n_sigma)
ax.set_ylim(0,n_nN*n_sigma)
ax.xaxis.set_major_locator(MultipleLocator(n_sigma))
ax.yaxis.set_major_locator(MultipleLocator(n_sigma))
ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.yaxis.set_minor_locator(MultipleLocator(1))
nN_tick_pos = [i*n_sigma for i,_ in enumerate(nN)]
ax.set_xticks(nN_tick_pos,nN)
ax.set_yticks(nN_tick_pos,nN)
ax.grid(True,which='both')
ax.set_xlabel('Nearest Neighbour')
ax.set_ylabel('Nearest Neighbour')
ax.set_title(r'Heat Map of nN and $\sigma_{factor}$')
#ax.minorticks_on()
fig.colorbar(im,ticks=np.linspace(0,nEigs+1,4))


plt.savefig('test_heatmap.png')


