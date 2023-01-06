#!/bin/bash
export MASTER_ADDR=$(hostname)
image=nersc/pytorch:ngc-22.02-v0
ngpu=4
config_file=./config/AFNO.yaml
config="afno_backbone_era5res_p4_e768_depth12_lr1em3"
run_num="check"
cmd="python train_cmip.py --enable_amp --yaml_config=$config_file --config=$config --run_num=$run_num"
srun -n $ngpu --cpus-per-task=32 --gpus-per-node $ngpu shifter --image=${image} bash -c "source export_DDP_vars.sh && $cmd"
