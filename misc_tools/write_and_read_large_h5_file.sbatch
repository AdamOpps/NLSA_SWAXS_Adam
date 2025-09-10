#!/bin/bash
sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=write_and_read_large_h5_file
#SBATCH --output=%x-%j.out
#SBATCH -N 1
#SBATCH --partition=cxfel
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=$USER"@uwm.edu"
python3 write_and_read_large_h5_file.py 1000
python3 write_and_read_large_h5_file.py 1000000
exit 0
EOT
