#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
shifter --image=nersc/pytorch:ngc-21.08-v1 python inference/inference.py --config=afno_backbone_25var_p2_e768_50km --run_num=0
#shifter --image=nersc/pytorch:ngc-21.08-v1 python inference/inference_precip.py --config=precip_inf --run_num=0 --weights=/global/cfs/cdirs/m4134/gsharing/model_weights/FCN_weights_v0/precip.ckpt --override_dir=/global/cfs/cdirs/dasrepo/shashank/fcn/precip --vis
