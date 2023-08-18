#!/bin/bash
config="swin_73var_p4_wr80_e768_d24_dpr01_lr1em3_abspos_roll_ft"
yaml=./config/era5_swin.yaml
run_num="0"
scratch="/pscratch/sd/s/shas1693/results/era5_wind/"
weights="${scratch}/${config}/${run_num}/training_checkpoints/best_ckpt.tar"

#weights="${scratch}/${config}/${run_num}/training_checkpoints/best_ckpt.tar"
#image=nersc/pytorch:ngc-23.04-v0
#env=/global/homes/s/shas1693/.local/perlmutter/nersc_pytorch_ngc-23.04-v0

image=nersc/pytorch:ngc-23.03-v0
env=$HOME/.local/perlmutter/nersc-pytorch-23.03-v0/

export CUDA_VISIBLE_DEVICES=0
year=2018_long
override_dir="${scratch}/${config}/year${year}/"
shifter --image=${image} --env PYTHONUSERBASE=${env} bash -c "python inference/inference.py --yaml_config=${yaml} --config=${config} --weights=${weights} --override_dir=${override_dir}"

#shifter --image=${image} nohup python inference/inference_hrmip.py --config=${config} --weights=${weights} --override_dir=${override_dir} --year=${year} --save &
#ngpu=4
#srun -l -n $ngpu --cpus-per-task=32 --gpus-per-node $ngpu shifter --image=${image} bash -c "$cmd"
