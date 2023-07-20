#!/bin/bash
config="sfno_73ch"
yaml=./config/cos_zenith_sfnonet.yaml
run_num="0"
scratch="/pscratch/sd/s/shas1693/results/sfno/"
weights="${scratch}/sfno_73ch_modulus/${run_num}/training_checkpoints/best_ckpt.tar"

#weights="${scratch}/${config}/${run_num}/training_checkpoints/best_ckpt.tar"
#image=nersc/pytorch:ngc-23.04-v0
#env=/global/homes/s/shas1693/.local/perlmutter/nersc_pytorch_ngc-23.04-v0

image=amahesh19/fcn-mip:0.5
env=/global/homes/s/shas1693/.local/perlmutter/amahesh19_fcnmip_0.5

export CUDA_VISIBLE_DEVICES=0
year=2018
override_dir="${scratch}/${config}/year${year}/"
shifter --image=${image} --env PYTHONUSERBASE=${env} bash -c "python inference/inference_sfno.py --yaml_config=${yaml} --config=${config} --weights=${weights} --override_dir=${override_dir}"

#shifter --image=${image} nohup python inference/inference_hrmip.py --config=${config} --weights=${weights} --override_dir=${override_dir} --year=${year} --save &
#ngpu=4
#srun -l -n $ngpu --cpus-per-task=32 --gpus-per-node $ngpu shifter --image=${image} bash -c "$cmd"
