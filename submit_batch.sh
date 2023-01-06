#!/bin/bash -l
#SBATCH --time=06:00:00
#SBATCH -C gpu
#SBATCH --account=m1517
#SBATCH -q regular
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH -J p2_e768
#SBATCH --image=nersc/pytorch:ngc-22.02-v0
#SBATCH --module=gpu,nccl-2.15
#SBATCH -o afno_backbone_25var_lamb_p2_e768_depth12_lr1em3_50km_0.out

config_file=./config/AFNO.yaml
config='afno_backbone_25var_lamb_p2_e768_depth12_lr1em3_50km'
run_num='nccl'

export HDF5_USE_FILE_LOCKING=FALSE
export NCCL_NET_GDR_LEVEL=PHB

export MASTER_ADDR=$(hostname)

set -x
srun -u --mpi=pmi2 shifter \
    bash -c "
    source export_DDP_vars.sh
    python train.py --enable_amp --yaml_config=$config_file --config=$config --run_num=$run_num
    "
