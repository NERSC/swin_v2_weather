#!/bin/bash -l
#SBATCH --time=06:00:00
#SBATCH -C gpu
#SBATCH --account=m1517
#SBATCH -q regular
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH -J fcn_dev
#SBATCH --image=nersc/pytorch:ngc-22.02-v0
#SBATCH -o v9xok1kp_%j.out

config_file=./config/AFNO.yaml
config='afno_backbone_25var_50km'
id="v9xok1kp"

export HDF5_USE_FILE_LOCKING=FALSE
export NCCL_NET_GDR_LEVEL=PHB
export CXI_FORK_SAFE=1
export CXI_FORK_SAFE_HP=1

export MASTER_ADDR=$(hostname)

cmd="python train.py --yaml_config=$config_file --config=$config --sweep_id=$id"

set -x
srun -u --mpi=pmi2 shifter --module gpu \
    bash -c "
    source export_DDP_vars.sh
    $cmd
    "
