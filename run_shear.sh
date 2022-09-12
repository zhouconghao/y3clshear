#!/bin/bash
#SBATCH --constraint=haswell
#SBATCH --nodes=3
#SBATCH --time=5

module purge
module load python
srun -n 96 -c 2  python run_cl.py