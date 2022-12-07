#!/bin/bash -l
#SBATCH --time=06:00:00
#SBATCH -C gpu
#SBATCH --account=nstaff_g
#SBATCH -q regular
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH -J afno_e300
#SBATCH --image=nersc/pytorch:ngc-22.02-v0
#SBATCH -o afno_backbone_26var_lamb_embed1536_dpr05_depth24_e300_dt4.out

config_file=./config/AFNO.yaml
config='afno_backbone_26var_lamb_embed1536_dpr05_depth24_e300_dt4'
run_num='0'

export HDF5_USE_FILE_LOCKING=FALSE
export NCCL_NET_GDR_LEVEL=PHB

export MASTER_ADDR=$(hostname)

set -x
srun -u --mpi=pmi2 shifter \
    bash -c "
    source export_DDP_vars.sh
    python train.py --enable_amp --yaml_config=$config_file --config=$config --run_num=$run_num
    "
