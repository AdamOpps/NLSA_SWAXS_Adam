################################################################################
def write_run_info(data_file,variable_name,N,D,measure_type,c,\
  h5=True,transpose=False):
# 
# copyright (c) Russell Fung 2020
# updated May 11, 2022
################################################################################
  
  run_info = open(".sna_run_info","w")
  run_info.write(data_file+'\n'+variable_name+'\n'+str(N)+'\n'+str(D)+'\n'+
    measure_type+'\n'+str(c)+'\n'+str(h5)+'\n'+str(transpose)+'\n')
  run_info.close()

