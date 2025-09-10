# add cxfel code to system path.
import os

cxfel_root = os.environ['CXFEL_ROOT']
startup_file = cxfel_root+'/misc_tools/startup.py'
exec(open(startup_file).read())

from mpi4py import MPI

comm = MPI.COMM_WORLD
my_rank = comm.Get_rank()
group_size = comm.Get_size()

from misc_tools import mpi_release_turn,mpi_request_turn

get_node_spec_info = 1000

mpi_request_turn(comm,taskID=get_node_spec_info)

os.system('echo hostname:')
os.system('hostname')
os.system('echo cpu:')
os.system('cat /proc/cpuinfo | grep model\ name | uniq')
os.system('cat /proc/cpuinfo | grep processor | wc -l')
os.system('echo gpu:')
os.system('lspci -v | grep 3D\ controller:\ NVIDIA\ Corporation')
os.system('ls -l /usr/lib64/libcuda.so')
os.system('echo ')

mpi_release_turn(comm,taskID=get_node_spec_info)

