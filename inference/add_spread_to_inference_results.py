#BSD 3-Clause License
#
#Copyright (c) 2022, FourCastNet authors
#All rights reserved.
#
#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:
#
#1. Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
#2. Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
#3. Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#The code was authored by the following people:
#
#Jaideep Pathak - NVIDIA Corporation
#Shashank Subramanian - NERSC, Lawrence Berkeley National Laboratory
#Peter Harrington - NERSC, Lawrence Berkeley National Laboratory
#Sanjeev Raja - NERSC, Lawrence Berkeley National Laboratory 
#Ashesh Chattopadhyay - Rice University 
#Morteza Mardani - NVIDIA Corporation 
#Thorsten Kurth - NVIDIA Corporation 
#David Hall - NVIDIA Corporation 
#Zongyi Li - California Institute of Technology, NVIDIA Corporation 
#Kamyar Azizzadenesheli - Purdue University 
#Pedram Hassanzadeh - Rice University 
#Karthik Kashinath - NVIDIA Corporation 
#Animashree Anandkumar - California Institute of Technology, NVIDIA Corporation

import os
import sys
import time
import numpy as np
import argparse
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../')
from numpy.core.numeric import False_
import h5py
import pdb
import torch
from torchvision.utils import save_image
import torch.nn as nn
import torch.cuda.amp as amp
import torch.distributed as dist
from collections import OrderedDict
from torch.nn.parallel import DistributedDataParallel
import logging
from utils import logging_utils
from utils.weighted_acc_rmse import weighted_spread, weighted_rmse_torch_channels, weighted_acc_torch_channels, unweighted_acc_torch_channels, weighted_acc_masked_torch_channels
logging_utils.config_logger()
from utils.YParams import YParams
from utils.data_loader_multifiles import get_data_loader
from networks.afno import AFNONet
from networks.swinv2 import swinv2net
#from networks.afnonet_decoder import AFNONet, PrecipNet
import wandb
import matplotlib.pyplot as plt
import glob
import datetime
from skimage.transform import downscale_local_mean
from modulus.utils.sfno.zenith_angle import cos_zenith_angle
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap as ruamelDict



fld = "z500" # just used for log to sceen


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_config", default='./config/AFNO.yaml', type=str)
    parser.add_argument("--config", default='full_field', type=str)
    parser.add_argument("--override_dir", default=None, type = str, help = 'Path to store inference outputs; must also set --weights arg')
    parser.add_argument("--weights", default=None, type=str, help = 'Path to model weights, for use with override_dir option')
    parser.add_argument("--save_jit", action='store_true')
    
    args = parser.parse_args()
    params = YParams(os.path.abspath(args.yaml_config), args.config)

    # Set up directory
    if args.override_dir is not None:
      expDir = args.override_dir
    else: 
       raise Exception("override directory needed")

    if not os.path.isdir(expDir):
      os.makedirs(expDir)

    params['experiment_dir'] = os.path.abspath(expDir)
    n_ics = params['n_initial_conditions']

    try:
      autoregressive_inference_filetag = params["inference_file_tag"]
    except:
      autoregressive_inference_filetag = ""


    autoregressive_inference_filetag += "_" + fld + ""

    sample_data = np.load(expDir + "pred_ic0.npy")
    _, n_timesteps, n_channels, h, w = sample_data.shape

    # Path for the HDF5 file to store the weighted spread
    prediction_file = os.path.join(params['experiment_dir'], 'autoregressive_predictions' + autoregressive_inference_filetag + '.h5')

    # Prepare the HDF5 file
    with h5py.File(prediction_file, 'a') as f:
        # If "spread" dataset exists, delete it to avoid issues
        if "spread" in f:
            del f["spread"]

        # Create the "spread" dataset with appropriate shape
        spread_dataset = f.create_dataset("spread", shape=(n_timesteps, n_channels), dtype='float64')

        for t in range(n_timesteps):
            print("timestep ",t,"/",n_timesteps)
            timestep_pred = np.zeros((n_ics, n_channels, h, w), dtype='float64')  # Adjusted to 4D

            # Load the data for the current timestep from all ICs
            for i in range(n_ics):
                print("ic ",i,'/',n_ics)
                pred = np.load(expDir + f"pred_ic{i}.npy")
                timestep_pred[i] = pred[0,t]  # Corrected assignment

            # Now we need to add the extra dimension back for weighted_spread
            timestep_pred = timestep_pred[:, np.newaxis, :, :, :]  # Adding the singleton timestep dimension

            # Compute the weighted spread for the current timestep
            weighted_spread_timestep = weighted_spread(timestep_pred)

            # Store the result in the HDF5 file for the current timestep
            spread_dataset[t, :] = weighted_spread_timestep.squeeze()
        f.close()
