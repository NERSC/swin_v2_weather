#!/bin/bash -l
#SBATCH --time=06:00:00
#SBATCH -C gpu
#SBATCH --account=m4331
#SBATCH -q regular
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH -J cmip
#SBATCH --image=nersc/pytorch:ngc-22.02-v0
#SBATCH --module=gpu,nccl-2.15
#SBATCH -o afno_backbone_ec3p_r2i1p1f1_p4_e768_depth12_lr1em3_finetune_ete150_0.out

config_file=./config/hrmip.yaml
config='afno_backbone_ec3p_r2i1p1f1_p4_e768_depth12_lr1em3_finetune_ete150'
run_num='0'

export HDF5_USE_FILE_LOCKING=FALSE
export NCCL_NET_GDR_LEVEL=PHB

export MASTER_ADDR=$(hostname)

set -x
srun -u --mpi=pmi2 shifter \
    bash -c "
    source export_DDP_vars.sh
    python train_hrmip.py --enable_amp --yaml_config=$config_file --config=$config --run_num=$run_num
    "
