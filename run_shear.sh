#!/bin/bash
#SBATCH --constraint=cpu
#SBATCH --nodes=1
#SBATCH --time=05:00:00
#SBATCH --account=des
#SBATCH --qos=regular

pwd; hostname; date
echo "Running program on $SLURM_JOB_NUM_NODES nodes with $SLURM_NTASKS total tasks, with each node getting $SLURM_NTASKS_PER_NODE running on cores."

module unload python
module load python

which python
which conda

source activate y3clshear
echo $CONDA_PREFIX

#srun -n 10 python mympi.py
srun -n 1 python -u run_cl.py 0.1 0.5 2 /global/cfs/cdirs/des/zhou/y3clshear/catalogs 20 100 0.1 10 10 dnf