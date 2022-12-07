#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
shifter --image=nersc/pytorch:ngc-21.08-v1 python inference/inference.py --config=afno_backbone_26var_lamb_embed1536_dpr03_dt4_newstats --run_num=0
#shifter --image=nersc/pytorch:ngc-21.08-v1 python inference/inference_precip.py --config=precip_inf --run_num=0 --weights=/global/cfs/cdirs/m4134/gsharing/model_weights/FCN_weights_v0/precip.ckpt --override_dir=/global/cfs/cdirs/dasrepo/shashank/fcn/precip --vis
