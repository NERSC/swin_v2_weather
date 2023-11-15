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
import torch
import torch.nn as nn
import torch.cuda.amp as amp
import torch.distributed as dist
from collections import OrderedDict
import logging
from utils import logging_utils
from utils.weighted_acc_rmse import weighted_rmse_torch_channels, weighted_acc_torch_channels, unweighted_acc_torch_channels, weighted_acc_masked_torch_channels
logging_utils.config_logger()
from utils.YParams import YParams
from utils.data_loader_multifiles import get_data_loader
from networks.afno import AFNONet
from networks.swinv2 import swinv2net
#from networks.afnonet_decoder import AFNONet, PrecipNet
import glob
import datetime
from skimage.transform import downscale_local_mean
from modulus.utils.sfno.zenith_angle import cos_zenith_angle
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap as ruamelDict



fld = "z500" # just used for log to screen

def compute_zenith_angle(local_idx, year, lon_grid, lat_grid, dt=1, n_history=0):
    # compute hours into the year
    jan_01_epoch = datetime.datetime(year, 1, 1, 0, 0, 0)

    # zenith angle for input
    cos_zenith_inp = []
    for idx in range(local_idx - dt * n_history, local_idx + 1, dt):
        hours_since_jan_01 = idx * 6
        model_time = jan_01_epoch + datetime.timedelta(hours=hours_since_jan_01)
        cos_zenith_inp.append(
            cos_zenith_angle(
                model_time, lon_grid, lat_grid
            ).astype(np.float32)
        )

    cos_zenith_inp = np.stack(cos_zenith_inp, axis=0)
    return torch.as_tensor(cos_zenith_inp)

def downscale(img, scale):
    new_img = downscale_local_mean(img, (1, 1, scale[0], scale[1]))
    return new_img

def gaussian_perturb(x, level=0.01, device=0):
    noise = level * torch.randn(x.shape).to(device, dtype=torch.float)
    return (x + noise)

def load_model(model, params, checkpoint_file):
    model.zero_grad()
    checkpoint_fname = checkpoint_file
    checkpoint = torch.load(checkpoint_fname)
    try:
        new_state_dict = OrderedDict()
        for key, val in checkpoint['model_state'].items():
            name = key[7:]
            if name != 'ged':
                new_state_dict[name] = val  
        model.load_state_dict(new_state_dict)
    except:
        model.load_state_dict(checkpoint['model_state'])
    model.eval()
    return model

def downsample(x, scale=0.125):
    return torch.nn.functional.interpolate(x, scale_factor=scale, mode='bilinear')

def setup(params, save_jit, expDir):
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    #get data loader
    _, valid_dataset = get_data_loader(params, params.inf_data_path, dist.is_initialized(), train=False)
    img_shape_x = valid_dataset.img_shape_x
    img_shape_y = valid_dataset.img_shape_y
    params.img_shape_x = img_shape_x
    params.img_shape_y = img_shape_y
    if params.log_to_screen:
        logging.info('Loading trained model checkpoint from {}'.format(params['best_checkpoint_path']))

    in_channels = np.array(params.in_channels)
    out_channels = np.array(params.out_channels)
    n_in_channels = len(in_channels)
    n_out_channels = len(out_channels)
    
    if params["orography"] or params['add_zenith']:
      params['N_in_channels'] = n_in_channels + 1
    else:
      params['N_in_channels'] = n_in_channels
    params['N_out_channels'] = n_out_channels

    # Save out config
    hparams = ruamelDict()
    yaml = YAML()
    for key, value in params.params.items():
        hparams[str(key)] = value
        with open(os.path.join(expDir, 'hyperparams.yaml'), 'w') as hpfile:
            yaml.dump(hparams, hpfile)
    params.means = np.load(params.global_means_path)[0, out_channels] # needed to standardize wind data
    params.stds = np.load(params.global_stds_path)[0, out_channels]
    params.device = device

    # load the model
    if params.nettype == 'afno':
      model = AFNONet(params).to(device) 
    elif params.nettype == 'swin':
      if save_jit:
        model = swinv2net(params, checkpoint_stages=False).to(device)
        script = torch.jit.script(model)
        torch.jit.save(script, expDir+'torchscripted.pt')
      else:
        model = swinv2net(params, checkpoint_stages=False).to(device)
    else:
      raise Exception("not implemented")

    checkpoint_file  = params['best_checkpoint_path']
    model = load_model(model, params, checkpoint_file)
    model = model.to(device)
    if train_mode:
       model.train()

    # load the validation data
    files_paths = glob.glob(params.inf_data_path + "/*.h5")
    files_paths.sort()
    # which year
    yr = 0
    if params.log_to_screen:
        logging.info('Loading inference data')
        logging.info('Inference data from {}'.format(files_paths[yr]))

    valid_data_full = h5py.File(files_paths[yr], 'r')['fields']


    return valid_data_full, model, files_paths[yr]

def autoregressive_inference(params, start_ind, valid_data_full, model, year=2018): 
    start_ind = int(start_ind) 
    #initialize global variables
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    dt = int(params.dt)
    prediction_length = int(params.prediction_length/dt)
    n_history = params.n_history
    img_shape_x = params.img_shape_x
    img_shape_y = params.img_shape_y
    in_channels = np.array(params.in_channels)
    out_channels = np.array(params.out_channels)
    n_in_channels = len(in_channels)
    n_out_channels = len(out_channels)
    means = params.means
    stds = params.stds

    #initialize memory for image sequences and RMSE/ACC
    valid_loss = torch.zeros((prediction_length, n_out_channels)).to(device, dtype=torch.float)
    acc = torch.zeros((prediction_length, n_out_channels)).to(device, dtype=torch.float)

    # compute metrics in a coarse resolution too if params.interp is nonzero
    valid_loss_coarse = torch.zeros((prediction_length, n_out_channels)).to(device, dtype=torch.float)
    acc_coarse = torch.zeros((prediction_length, n_out_channels)).to(device, dtype=torch.float)
    acc_coarse_unweighted = torch.zeros((prediction_length, n_out_channels)).to(device, dtype=torch.float)
    
    acc_unweighted = torch.zeros((prediction_length, n_out_channels)).to(device, dtype=torch.float)
    seq_real = torch.zeros((prediction_length, n_in_channels, img_shape_x, img_shape_y)).to(device, dtype=torch.float)
    seq_pred = torch.zeros((prediction_length, n_in_channels, img_shape_x, img_shape_y)).to(device, dtype=torch.float)

    acc_land = torch.zeros((prediction_length, n_out_channels)).to(device, dtype=torch.float)
    acc_sea = torch.zeros((prediction_length, n_out_channels)).to(device, dtype=torch.float)
    if params.masked_acc:
      maskarray = torch.as_tensor(np.load(params.maskpath)[0:720]).to(device, dtype=torch.float)
    valid_data = valid_data_full[start_ind:(start_ind+prediction_length*dt+n_history*dt):dt, in_channels, 0:720] #extract valid data from first year
    if params.interp_factor_x != 1 or params.interp_factor_y != 1:
        valid_data = downscale(valid_data, scale = (params.interp_factor_x, params.interp_factor_y))
    # standardize
    valid_data = (valid_data - means)/stds
    valid_data = torch.as_tensor(valid_data).to(device, dtype=torch.float)

    #load time means
    if not params.use_daily_climatology:
      m = torch.as_tensor((np.load(params.time_means_path)[0][out_channels] - means)/stds)[:, 0:img_shape_x] # climatology
      m = torch.unsqueeze(m, 0)
    else:
      # use daily clim like weyn et al. (different from rasp)
      dc_path = params.dc_path
      with h5py.File(dc_path, 'r') as f:
        dc = f['time_means_daily'][start_ind:start_ind+prediction_length*dt:dt] # 1460,21,721,1440
      m = torch.as_tensor((dc[:,out_channels,0:img_shape_x,:] - means)/stds) 

    m = m.to(device, dtype=torch.float)
    if params.interp > 0:
        m_coarse = downsample(m, scale=params.interp)

    std = torch.as_tensor(stds[:,0,0]).to(device, dtype=torch.float)

    orography = params.orography
    orography_path = params.orography_path
    if orography:
      orog = torch.as_tensor(np.expand_dims(np.expand_dims(h5py.File(orography_path, 'r')['orog'][0:720], axis = 0), axis = 0)).to(device, dtype = torch.float)
      logging.info("orography loaded; shape:{}".format(orog.shape))

    if params.add_zenith:
        # we need some additional static fields in this case
        longitude = np.arange(0, 360, 0.25)
        latitude = np.arange(-90, 90.25, 0.25)
        latitude = latitude[::-1]
        lon_grid_local, lat_grid_local = np.meshgrid(longitude, latitude)
        

    #autoregressive inference
    if params.log_to_screen:
      logging.info('Begin autoregressive inference')
    with torch.no_grad():
      for i in range(valid_data.shape[0]): 
        if i==0: #start of sequence
          first = valid_data[0:n_history+1]
          future = valid_data[n_history+1]
          for h in range(n_history+1):
            seq_real[h] = first[h*n_in_channels : (h+1)*n_in_channels][0:n_out_channels] #extract history from 1st 
            seq_pred[h] = seq_real[h]
          if params.perturb:
            first = gaussian_perturb(first, level=params.n_level, device=device) # perturb the ic
          if orography:
            future_pred = model(torch.cat((first, orog), axis=1))
          elif params.add_zenith:
            zen_inp  = compute_zenith_angle(start_ind, year, lon_grid_local, lat_grid_local)
            zen_inp = zen_inp[:,:img_shape_x].to(device).unsqueeze(0)
            future_pred = model(torch.cat((first, zen_inp), axis=1))
          else:
            future_pred = model(first)
        else:
          if i < prediction_length-1:
            future = valid_data[n_history+i+1]
          if orography:
            future_pred = model(torch.cat((future_pred, orog), axis=1)) #autoregressive step
          elif params.add_zenith:
            zen_inp  = compute_zenith_angle(start_ind + i, year, lon_grid_local, lat_grid_local)
            zen_inp = zen_inp[:,:img_shape_x].to(device).unsqueeze(0)
            future_pred = model(torch.cat((future_pred, zen_inp), axis=1))
          else:
            future_pred = model(future_pred) #autoregressive step

        if i < prediction_length-1: #not on the last step
          seq_pred[n_history+i+1] = future_pred
          seq_real[n_history+i+1] = future
          history_stack = seq_pred[i+1:i+2+n_history]

        future_pred = history_stack
      
        #Compute metrics 
        if params.use_daily_climatology:
            clim = m[i:i+1]
            if params.interp > 0:
                clim_coarse = m_coarse[i:i+1]
        else:
            clim = m
            if params.interp > 0:
                clim_coarse = m_coarse

        pred = torch.unsqueeze(seq_pred[i], 0)
        tar = torch.unsqueeze(seq_real[i], 0)
        valid_loss[i] = weighted_rmse_torch_channels(pred, tar) * std
        # spread[i] = weighted_spread_torch_channels(pred)

        acc[i] = weighted_acc_torch_channels(pred-clim, tar-clim)
        acc_unweighted[i] = unweighted_acc_torch_channels(pred-clim, tar-clim)

        if params.masked_acc:
          acc_land[i] = weighted_acc_masked_torch_channels(pred-clim, tar-clim, maskarray)
          acc_sea[i] = weighted_acc_masked_torch_channels(pred-clim, tar-clim, 1-maskarray)

        if params.interp > 0:
            pred = downsample(pred, scale=params.interp)
            tar = downsample(tar, scale=params.interp)
            valid_loss_coarse[i] = weighted_rmse_torch_channels(pred, tar) * std
            acc_coarse[i] = weighted_acc_torch_channels(pred-clim_coarse, tar-clim_coarse)
            acc_coarse_unweighted[i] = unweighted_acc_torch_channels(pred-clim_coarse, tar-clim_coarse)

        if params.log_to_screen:
          idx = params.channel_names.index(fld)
          logging.info('Predicted timestep {} of {}. {} RMS Error: {:.2f}, ACC: {:.2f}'.format(i, prediction_length, fld, valid_loss[i, idx], acc[i, idx]))
          if params.interp > 0:
            logging.info('[COARSE] Predicted timestep {} of {}. {} RMS Error: {}, ACC: {}'.format(i, prediction_length, fld, valid_loss_coarse[i, idx],
                        acc_coarse[i, idx]))

    seq_real = seq_real.cpu().numpy()
    seq_pred = seq_pred.cpu().numpy()
    valid_loss = valid_loss.cpu().numpy()
    # spread = spread.cpu().numpy()
    acc = acc.cpu().numpy()
    acc_unweighted = acc_unweighted.cpu().numpy()
    acc_coarse = acc_coarse.cpu().numpy()
    acc_coarse_unweighted = acc_coarse_unweighted.cpu().numpy()
    valid_loss_coarse = valid_loss_coarse.cpu().numpy()
    acc_land = acc_land.cpu().numpy()
    acc_sea = acc_sea.cpu().numpy()

    return (np.expand_dims(seq_real[n_history:], 0), np.expand_dims(seq_pred[n_history:], 0), np.expand_dims(valid_loss,0), np.expand_dims(acc, 0),
           np.expand_dims(acc_unweighted, 0), np.expand_dims(valid_loss_coarse, 0), np.expand_dims(acc_coarse, 0),
           np.expand_dims(acc_coarse_unweighted, 0),
           np.expand_dims(acc_land, 0),
           np.expand_dims(acc_sea, 0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_num", default='00', type=str)
    parser.add_argument("--yaml_config", default='./config/AFNO.yaml', type=str)
    parser.add_argument("--config", default='full_field', type=str)
    parser.add_argument("--use_daily_climatology", action='store_true')
    parser.add_argument("--vis", action='store_true')
    parser.add_argument("--include_spread", action='store_true')
    parser.add_argument("--train_mode", action='store_true')
    parser.add_argument("--override_dir", default=None, type = str, help = 'Path to store inference outputs; must also set --weights arg')
    parser.add_argument("--interp", default=0, type=float)
    parser.add_argument("--doy_start", default=0, type=int)
    parser.add_argument("--weights", default=None, type=str, help = 'Path to model weights, for use with override_dir option')
    parser.add_argument("--save_jit", action='store_true')
    
    args = parser.parse_args()
    params = YParams(os.path.abspath(args.yaml_config), args.config)
    params['world_size'] = 1
    params['interp'] = args.interp
    params['use_daily_climatology'] = args.use_daily_climatology
    params['global_batch_size'] = params.batch_size

    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = True
    vis = args.vis
    include_spread = args.include_spread
    train_mode = args.train_mode
    doy_start = args.doy_start

    # Set up directory
    if args.override_dir is not None:
      assert args.weights is not None, 'Must set --weights argument if using --override_dir'
      expDir = args.override_dir
    else:
      assert args.weights is None, 'Cannot use --weights argument without also using --override_dir'
      expDir = os.path.join(params.exp_dir, args.config, str(args.run_num))

    if not os.path.isdir(expDir):
      os.makedirs(expDir)

    params['experiment_dir'] = os.path.abspath(expDir)
    params['best_checkpoint_path'] = args.weights if args.override_dir is not None else os.path.join(expDir, 'training_checkpoints/best_ckpt.tar')
    params['resuming'] = False
    params['local_rank'] = 0

    logging_utils.log_to_file(logger_name=None, log_filename=os.path.join(expDir, 'inference_out.log'))
    logging_utils.log_versions()
    params.log()

    n_ics = params['n_initial_conditions']

    if fld == "z500" or fld == "t850":
        n_samples_per_year = 1336
    else:
        n_samples_per_year = 1460

    logging.info("Inference for {} initial conditions".format(n_ics))
    try:
      autoregressive_inference_filetag = params["inference_file_tag"]
    except:
      autoregressive_inference_filetag = ""

    if params.interp > 0:
        autoregressive_inference_filetag = "_coarse"

    autoregressive_inference_filetag += "_" + fld + ""
    if vis:
        autoregressive_inference_filetag += "_vis"

    # get data and models
    valid_data_full, model, data_path = setup(params, args.save_jit, expDir)
    year = int(os.path.basename(data_path).replace('.h5', ''))

    #initialize lists for image sequences and RMSE/ACC
    valid_loss = []
    valid_loss_coarse = []
    acc_unweighted = []
    acc = []
    acc_coarse = []
    acc_coarse_unweighted = []
    seq_pred = []
    seq_real = []
    acc_land = []
    acc_sea = []
    spread = []

    #run autoregressive inference for multiple initial conditions
    for i in range(n_ics):
      logging.info("Initial condition {} of {}, year={}".format(i+1, n_ics, year))

      #set seed for random dropout variation
      torch.manual_seed(i)

      #multiply day of year by 4 for 6-hour timesteps to get start index
      start_ind = int(doy_start*4)
      sr, sp, vl, a, _, _, _, _, _, _ = autoregressive_inference(params, start_ind, valid_data_full, model, year=year)

      #if doing spread calculation save prediction for a given initial condition (crashes memory trying to accumulate here)
      if include_spread:
        np.save(expDir + "pred_ic"+str(i),sp)
      if i ==0 or len(valid_loss) == 0:
        seq_real = sr
        seq_pred = sp
        valid_loss = vl
        acc = a
      else:
        valid_loss = np.concatenate((valid_loss, vl), 0)
        acc = np.concatenate((acc, a), 0)

    prediction_length = seq_real[0].shape[0]
    n_out_channels = seq_real[0].shape[1]
    img_shape_x = seq_real[0].shape[2]
    img_shape_y = seq_real[0].shape[3]

    #save predictions and loss
    if params.log_to_screen:
      logging.info("Saving files at {}".format(os.path.join(params['experiment_dir'], 'autoregressive_predictions' + autoregressive_inference_filetag + '.h5')))
    with h5py.File(os.path.join(params['experiment_dir'], 'autoregressive_predictions'+ autoregressive_inference_filetag +'.h5'), 'a') as f:
      try:
          f.create_dataset("ground_truth", data = seq_real, shape = (1, prediction_length, n_out_channels, img_shape_x, img_shape_y), dtype = np.float32)
      except: 
          del f["ground_truth"]
          f.create_dataset("ground_truth", data = seq_real, shape = (1, prediction_length, n_out_channels, img_shape_x, img_shape_y), dtype = np.float32)
          f["ground_truth"][...] = seq_real
        # try:
        #     f.create_dataset("predicted", data = seq_pred, shape = (n_ics, prediction_length, n_out_channels, img_shape_x, img_shape_y), dtype = np.float32)
        # except:
        #     del f["predicted"]
        #     f.create_dataset("predicted", data = seq_pred, shape = (n_ics, prediction_length, n_out_channels, img_shape_x, img_shape_y), dtype = np.float32)
        #     f["predicted"][...]= seq_pred
      try:
        f.create_dataset("rmse", data = valid_loss, shape = (n_ics, prediction_length, n_out_channels), dtype =np.float32)
      except:
        del f["rmse"]
        f.create_dataset("rmse", data = valid_loss, shape = (n_ics, prediction_length, n_out_channels), dtype =np.float32)
        f["rmse"][...] = valid_loss

      try:
        f.create_dataset("acc", data = acc, shape = (n_ics, prediction_length, n_out_channels), dtype =np.float32)
      except:
        del f["acc"]
        f.create_dataset("acc", data = acc, shape = (n_ics, prediction_length, n_out_channels), dtype =np.float32)
        f["acc"][...] = acc   

      f.close()