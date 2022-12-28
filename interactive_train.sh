#!/bin/bash
export MASTER_ADDR=$(hostname)
image=nersc/pytorch:ngc-22.02-v0
ngpu=4
config_file=./config/AFNO.yaml
config="afno_backbone_25var_50km"
run_num="check"
cmd="python train.py --enable_amp --yaml_config=$config_file --config=$config --run_num=$run_num"
srun -n $ngpu --cpus-per-task=32 --gpus-per-node $ngpu shifter --image=${image} bash -c "source export_DDP_vars.sh && $cmd"
