#!/bin/bash
export MASTER_ADDR=$(hostname)
image=nersc/pytorch:ngc-23.07-v0
env=~/.local/perlmutter/nersc_pytorch_ngc-23.07-v0
ngpu=4
config_file=./config/sfno.yaml
config="sfno_73ch"
run_num="check"
cmd="python train.py --enable_amp --yaml_config=$config_file --config=$config --run_num=$run_num"
srun -n $ngpu --cpus-per-task=32 --gpus-per-node $ngpu shifter --image=${image} --env PYTHONUSERBASE=$env bash -c "source export_DDP_vars.sh && $cmd"
