#!/bin/bash
export MASTER_ADDR=$(hostname)
image=nersc/pytorch:ngc-23.03-v0
ngpu=4
config_file=./config/era5_swin.yaml
config="swin_73var_p4_wr80_e768_d24_dpr01_lr1em3_roll_ft"
run_num="check"
cmd="python train.py --enable_amp --yaml_config=$config_file --config=$config --run_num=$run_num"
srun -n $ngpu --cpus-per-task=32 --gpus-per-node $ngpu shifter --image=${image} --env PYTHONUSERBASE=~/.local/perlmutter/nersc-pytorch-23.03-v0 bash -c "source export_DDP_vars.sh && $cmd"
