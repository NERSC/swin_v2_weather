#!/bin/bash -l
#SBATCH --time=06:00:00
#SBATCH -C 'gpu&hbm80g'
#SBATCH --account=m4416
#SBATCH -q regular
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --image=nersc/pytorch:ngc-23.07-v0
#SBATCH --module=gpu,nccl-2.18
#SBATCH -J swin-dali
#SBATCH -o %x-%j.out

config_file=./config/swin.yaml
config=$1
run_num='00'

export FI_MR_CACHE_MONITOR=userfaultfd
export HDF5_USE_FILE_LOCKING=FALSE
export NCCL_NET_GDR_LEVEL=PHB

export MASTER_ADDR=$(hostname)
# Reversing order of GPUs to match default CPU affinities from Slurm
export CUDA_VISIBLE_DEVICES=3,2,1,0

env=~/.local/perlmutter/nersc_pytorch_ngc-23.07-v0

set -x
srun -u shifter --env PYTHONUSERBASE=$env \
    bash -c "
    source export_DDP_vars.sh
    python train.py --enable_amp --yaml_config=$config_file --config=$config --run_num=$run_num
    "
