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

import logging
import glob
import torch
import random
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch import Tensor
import h5py
import math
#import cv2
from utils.img_utils import get_fields, interpolate

def get_metadata(pl_varlist, sl_varlist, level_list):

    base_dataset_pl_vars = OrderedDict(zip(['q', 't', 'u', 'v', 'z'], range(5)))
    base_dataset_sl_vars = OrderedDict(zip(['msl', 't2m', 'u10', 'v10'], range(4)))
    base_dataset_pressure_levels = OrderedDict(zip([925, 800, 700, 600, 500, 250], range(6)))

    pl_idx = [ base_dataset_pl_vars[pl_var] for pl_var in pl_varlist]
    sl_idx = [ base_dataset_sl_vars[sl_var] for sl_var in sl_varlist]
    level_idx = [ base_dataset_pressure_levels[pl] for pl in level_list ]

    return(pl_idx, sl_idx, level_idx)


def get_data_loader(params, files_pattern, distributed, split = "train"):

  train = (split == "train")
  dataset = GetDataset(params, files_pattern, split)
  sampler = DistributedSampler(dataset, shuffle=train) if distributed else None
  
  dataloader = DataLoader(dataset,
                          batch_size=int(params.batch_size),
                          num_workers=params.num_data_workers,
                          shuffle=False, #(sampler is None),
                          sampler=sampler if train else None,
                          drop_last=True,
                          pin_memory=torch.cuda.is_available())

  if train:
    return dataloader, dataset, sampler
  else:
    return dataloader, dataset

class GetDataset(Dataset):
  def __init__(self, params, location, split):
    self.params = params
    self.location = location
    self.train = (split == "train")
    self.split = split
    self.dt = params.dt
    self.n_history = params.n_history
    self.crop_size_x = params.crop_size_x
    self.crop_size_y = params.crop_size_y
    self.roll = params.roll

    self.interp_factor_x = params.interp_factor_x
    self.interp_factor_y = params.interp_factor_y
    self.interp_factor = (self.interp_factor_x, self.interp_factor_y)

    self._get_files_stats()

    self.two_step_training = params.two_step_training
    self.orography = params.orography
    self.add_noise = params.add_noise if self.train else False

    try:
        self.normalize = params.normalize
    except:
        self.normalize = True #by default turn on normalization if not specified in config

    if self.orography:
      self.orography_path = params.orography_path

  def _get_files_stats(self):
#    self.files_paths = glob.glob(self.location + "/*.h5")
#    self.files_paths.sort()

    if self.split == "train":
        years = list(range(self.params.train_years_range[0], self.params.train_years_range[1] + 1))
    elif self.split == "valid":
        years = list(range(self.params.valid_years_range[0], self.params.valid_years_range[1] + 1))
    elif self.split == "inference":
        years = list(range(self.params.test_years_range[0], self.params.test_years_range[1] + 1))

    self.files_paths = [self.location + "/{}.h5".format(yr) for yr in years]

    if hasattr(self.params, 'leave_out_years'):
        leave_out_years = [self.location + "/{}.h5".format(y) for y in self.params.leave_out_years]
        self.files_paths = [f for f in self.files_paths if f not in leave_out_years]

    if hasattr(self.params, 'use_years') and self.train:
        # overwrite with only use_years
        use_years = [self.location + "/{}.h5".format(y) for y in self.params.use_years]
        self.files_paths = use_years

    self.files_paths.sort()

    self.n_years = len(self.files_paths)
    stats_idx = 0
    # dont use leap year unless they are all leap years
    while int(self.files_paths[stats_idx].split("/")[-1].split(".")[0]) % 4 == 0:
        stats_idx += 1
        if stats_idx >= self.n_years:
            stats_idx = 0
            break

    with h5py.File(self.files_paths[stats_idx], 'r') as _f:
        logging.info("Getting file stats from {}".format(self.files_paths[0]))

        self.pl_shape = _f['pl'].shape[1]
        self.n_levels = _f['pl'].shape[2]
        self.sl_shape = _f['sl'].shape[1]

        self.n_samples_per_year = _f['pl'].shape[0]
        self.n_channels = self.pl_shape * self.n_levels + self.sl_shape

        self.img_shape_x = _f['sl'].shape[2]
        self.img_shape_y = _f['sl'].shape[3]

    self.n_samples_total = self.n_years * self.n_samples_per_year
    self.pl_files = [None for _ in range(self.n_years)]
    self.sl_files = [None for _ in range(self.n_years)]

    logging.info("Number of samples per year: {}".format(self.n_samples_per_year))
    logging.info("Found data at path {}. Number of examples: {}. Image Shape: {} x {} x {}".format(self.location, self.n_samples_total, self.img_shape_x, self.img_shape_y, self.n_channels))
    logging.info("Delta t: {} hours".format(6*self.dt))
    logging.info("Including {} hours of past history in training at a frequency of {} hours".format(6*self.dt*self.n_history, 6*self.dt))
    self.img_shape_x //= self.interp_factor_x
    self.img_shape_y //= self.interp_factor_y

  def _open_file(self, year_idx):
    _file = h5py.File(self.files_paths[year_idx], 'r')
    self.pl_files[year_idx] = _file['pl']  
    self.sl_files[year_idx] = _file['sl']  
    if self.orography:
      _orog_file = h5py.File(self.orography_path, 'r')
      self.orography_field = _orog_file['orog']
  
  def __len__(self):
    return self.n_samples_total

  def __getitem__(self, global_idx):
    year_idx = int(global_idx/self.n_samples_per_year) #which year we are on
    local_idx = int(global_idx%self.n_samples_per_year) #which sample in that year we are on - determines indices for centering

    #open image file
    if self.pl_files[year_idx] is None:
        self._open_file(year_idx)

    step = self.dt
    if self.two_step_training:
        local_idx = local_idx %(self.n_samples_per_year - 2*self.dt)
    else:
        local_idx = local_idx %(self.n_samples_per_year - self.dt)

    #if we are not at least self.dt*n_history timesteps into the prediction
    if local_idx < self.dt*self.n_history:
        local_idx += self.dt*self.n_history
 
    if self.two_step_training:
        pl_in  = self.pl_files[year_idx][(local_idx - self.dt * self.n_history):(local_idx + self.dt):self.dt]
        sl_in  = self.sl_files[year_idx][(local_idx - self.dt * self.n_history):(local_idx + self.dt):self.dt]
        pl_tar = self.pl_files[year_idx][(local_idx + step):(local_idx + step + 2*step):self.dt]
        sl_tar = self.sl_files[year_idx][(local_idx + step):(local_idx + step + 2*step):self.dt]
        inp, tar = get_fields(pl_in, sl_in, 'inp', self.params, self.train, self.add_noise), \
                    get_fields(pl_tar, sl_tar, 'tar', self.params, self.train) 
    else:
        pl_in  = self.pl_files[year_idx][(local_idx - self.dt * self.n_history):(local_idx + self.dt):self.dt]
        sl_in  = self.sl_files[year_idx][(local_idx - self.dt * self.n_history):(local_idx + self.dt):self.dt]
        pl_tar = self.pl_files[year_idx][local_idx + step]
        sl_tar = self.sl_files[year_idx][local_idx + step]
        inp, tar = get_fields(pl_in, sl_in, 'inp', self.params, self.train, self.add_noise), \
                    get_fields(pl_tar, sl_tar, 'tar', self.params, self.train) 

    if self.interp_factor_x != 1 or self.interp_factor_y != 1:
        inp, tar = interpolate(inp, tar, self.interp_factor)

    return inp, tar
