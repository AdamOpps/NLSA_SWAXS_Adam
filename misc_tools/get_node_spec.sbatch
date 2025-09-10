#!/bin/bash
sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=get_node_spec
#SBATCH --output=%x-%j.out
#SBATCH -N 2
#SBATCH --partition=batch
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=$USER"@uwm.edu"
mpiexec -N 2 python3 get_node_spec.py
exit 0
EOT
