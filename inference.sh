#!/bin/bash
#export CUDA_VISIBLE_DEVICES=0
#shifter --image=nersc/pytorch:ngc-21.08-v1 python inference/inference.py --config=afno_backbone_25var_lamb_p4_e768_depth12_lr1em3_50km_finetune --run_num=0
#shifter --image=nersc/pytorch:ngc-21.08-v1 python inference/inference_precip.py --config=precip_inf --run_num=0 --weights=/global/cfs/cdirs/m4134/gsharing/model_weights/FCN_weights_v0/precip.ckpt --override_dir=/global/cfs/cdirs/dasrepo/shashank/fcn/precip --vis
#shifter --image=nersc/pytorch:ngc-21.08-v1 python inference/inference_cmip.py --config=afno_backbone_cmip_p4_e768_depth12_lr1em3_finetune --run_num=era5res --weights=/pscratch/sd/s/shas1693/results/climate/afno_backbone_era5res_p4_e768_depth12_lr1em3_finetune/0/training_checkpoints/best_ckpt.tar --override_dir=/pscratch/sd/s/shas1693/results/climate/afno_backbone_cmip_p4_e768_depth12_lr1em3_finetune/era5res/ --vis
#shifter --image=nersc/pytorch:ngc-21.08-v1 python inference/inference_cmip.py --config=afno_backbone_cmip_p4_e768_depth12_lr1em3_finetune --run_num=era5res1

config="afno_backbone_ec3p_r2i1p1f1_p4_e768_depth12_lr1em3_finetune"
run_num="0"
scratch="/pscratch/sd/s/shas1693/results/climate/"
weights="${scratch}/${config}/${run_num}/training_checkpoints/best_ckpt.tar"
image=nersc/pytorch:ngc-22.02-v0

export CUDA_VISIBLE_DEVICES=0
year=2015
override_dir="${scratch}/${config}/year${year}/"
shifter --image=${image} nohup python inference/inference_hrmip.py --config=${config} --weights=${weights} --override_dir=${override_dir} --year=${year} --save &


#ngpu=4
#srun -l -n $ngpu --cpus-per-task=32 --gpus-per-node $ngpu shifter --image=${image} bash -c "$cmd"
