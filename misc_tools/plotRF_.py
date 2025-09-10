################################################################################
def plotRF(x,y,xlabel='x',ylabel='y',legend=[],legend_loc='lower right',\
  color=[],xticks=None,xticklabels=None,title='result',JPEG='result.jpg'):
# 
# copyright (c) Russell Fung 2022
################################################################################
  
  import socket
  if ("compute-" in socket.gethostname()):
    return
  
  import matplotlib.pyplot as plt
  
  n = len(x)
  
  if not color:
    color = []
    color.append('blue')
    for jj in range(1,n):
      color.append('green')
  if not legend:
    legend = []
    for jj in range(n):
      legend.append('curve '+str(jj))
  
  fig = plt.figure()
  for jj in range(n):
    plt.plot(x[jj],y[jj],c=color[jj],label=legend[jj])
  plt.xlabel(xlabel,fontsize=15)
  plt.ylabel(ylabel,fontsize=15)
  plt.title(title,fontsize=15)
  plt.legend(loc=legend_loc,fontsize=15)
  if xticks is not None:
    plt.xticks(ticks=xticks,labels=xticklabels)
  plt.show(block=False)
  plt.savefig(JPEG,bbox_inches='tight')

