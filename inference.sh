#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
#shifter --image=nersc/pytorch:ngc-21.08-v1 python inference/inference.py --config=afno_backbone_25var_lamb_p4_e768_depth12_lr1em3_50km_finetune --run_num=0
#shifter --image=nersc/pytorch:ngc-21.08-v1 python inference/inference_precip.py --config=precip_inf --run_num=0 --weights=/global/cfs/cdirs/m4134/gsharing/model_weights/FCN_weights_v0/precip.ckpt --override_dir=/global/cfs/cdirs/dasrepo/shashank/fcn/precip --vis
#shifter --image=nersc/pytorch:ngc-21.08-v1 python inference/inference_cmip.py --config=afno_backbone_cmip_p4_e768_depth12_lr1em3_finetune --run_num=era5res --weights=/pscratch/sd/s/shas1693/results/climate/afno_backbone_era5res_p4_e768_depth12_lr1em3_finetune/0/training_checkpoints/best_ckpt.tar --override_dir=/pscratch/sd/s/shas1693/results/climate/afno_backbone_cmip_p4_e768_depth12_lr1em3_finetune/era5res/
shifter --image=nersc/pytorch:ngc-21.08-v1 python inference/inference_cmip.py --config=afno_backbone_cmip_p4_e768_depth12_lr1em3_finetune --run_num=0 
