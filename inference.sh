#!/bin/bash
#config="afno_backbone_ec3p_r2i1p1f1_p4_e768_depth12_lr1em3_finetune"
config="afno_backbone_26var_finetune"
run_num="2"
scratch="/pscratch/sd/s/shas1693/results/era5_wind/"
#scratch="/pscratch/sd/s/shas1693/results/climate/"
weights="${scratch}/${config}/${run_num}/training_checkpoints/best_ckpt.tar"
image=nersc/pytorch:ngc-23.04-v0
env=/global/homes/s/shas1693/.local/perlmutter/nersc_pytorch_ngc-23.04-v0

export CUDA_VISIBLE_DEVICES=0
year=2018
override_dir="${scratch}/${config}/year${year}/"
shifter --image=${image} --env PYTHONUSERBASE=${env} bash -c "python inference/inference.py --config=${config} --weights=${weights} --override_dir=${override_dir}"

#shifter --image=${image} nohup python inference/inference_hrmip.py --config=${config} --weights=${weights} --override_dir=${override_dir} --year=${year} --save &



#ngpu=4
#srun -l -n $ngpu --cpus-per-task=32 --gpus-per-node $ngpu shifter --image=${image} bash -c "$cmd"
