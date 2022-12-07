#!/bin/bash
#SBATCH --time=03:00:00
#SBATCH -C haswell
#SBATCH -q regular
#SBATCH --account=nstaff
#SBATCH -N 19
#SBATCH --cpus-per-task=20
#SBATCH --ntasks-per-node=2
#SBATCH --image=nersc/pytorch:ngc-22.02-v0
#SBATCH -o regional.log

#module load python
#source activate lazy-h5py
#pip install --user scikit-image
export HDF5_USE_FILE_LOCKING=FALSE

srun -u --mpi=pmi2 shifter bash -c "python regional_boxes.py"
