import datetime
import glob
import h5py
import logging
from modulus.utils.sfno.zenith_angle import cos_zenith_angle
import numpy as np
import os
import random
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from typing import Tuple

def worker_init(wrk_id):
    np.random.seed(torch.utils.data.get_worker_info().seed%(2**32 - 1))

def is_leap_year(yr):
    return (yr%4 == 0)

def get_data_loader(params, files_pattern, distributed, train):
    dataset = GetDataset(params, files_pattern, train)

    if distributed:
        sampler = DistributedSampler(dataset, shuffle=train)
    else:
        sampler = None
    
    dataloader = DataLoader(dataset,
                            batch_size=int(params.local_batch_size),
                            num_workers=params.num_data_workers,
                            shuffle=(sampler is None),
                            sampler=sampler,
                            worker_init_fn=worker_init,
                            drop_last=True,
                            pin_memory=torch.cuda.is_available())

    if train:
        return dataloader, dataset, sampler
    else:
        return dataloader, dataset

class GetDataset(Dataset):
    def __init__(self, params, location, train):
        self.params = params
        self.location = location
        self.train = train
        self.dt = params.dt
        self.in_channels = params.in_channels
        self.out_channels = params.out_channels
        self.n_in_channels = params.n_in_channels
        self.n_out_channels = params.n_out_channels
        self.n_future = params.n_future
        self.normalize = True
        self.means = np.load(params.global_means_path)[0,self.in_channels]
        self.stds = np.load(params.global_stds_path)[0,self.in_channels]
        self._get_files_stats()
        if self.params.add_zenith:
            # additional static fields needed for coszen
            longitude = np.arange(0, 360, 0.25)
            latitude = np.arange(-90, 90.25, 0.25)
            latitude = latitude[::-1]
            self.lon_grid_local, self.lat_grid_local = np.meshgrid(longitude, latitude)

    def _get_files_stats(self):
        self.files_paths = glob.glob(self.location + "/*.h5")
        self.files_paths.sort()
        self.years = [int(os.path.splitext(os.path.basename(x))[0][-4:]) for x in self.files_paths]
        self.n_years = len(self.files_paths)

        # dont use leap year unless they are all leap years
        stats_idx = 0
        while is_leap_year(self.years[stats_idx]):
            stats_idx += 1
            if stats_idx >= self.n_years:
                stats_idx = 0
                break

        with h5py.File(self.files_paths[stats_idx], 'r') as _f:
            logging.info("Getting file stats from {}".format(self.files_paths[stats_idx]))
            self.n_samples_per_year = _f['fields'].shape[0]
            self.img_shape_x = self.params.img_size[0]
            self.img_shape_y = self.params.img_size[1]
            assert(self.img_shape_x <= _f['fields'].shape[2] and self.img_shape_y <= _f['fields'].shape[3]), 'image shapes are greater than dataset image shapes'

        self.n_samples_total = self.n_years * self.n_samples_per_year
        self.files = [None for _ in range(self.n_years)]
        logging.info("Number of samples per year: {}".format(self.n_samples_per_year))
        logging.info("Found data at path {}. Number of examples: {}. Image Shape: {} x {} x {}".format(self.location, self.n_samples_total, self.img_shape_x, self.img_shape_y, self.n_in_channels))
    
    def _open_file(self, year_idx):
        _file = h5py.File(self.files_paths[year_idx], 'r')
        self.files[year_idx] = _file['fields']  
    
    def __len__(self):
        return self.n_samples_total

    def _normalize(self, img):
        if self.normalize:
            img -= self.means
            img /= self.stds
        return torch.as_tensor(img)

    def _compute_zenith_angle(self, local_idx: int, year_idx: int, time_step_hours: int = 6) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the cosine of the zenith angle for specific time points.

        Parameters:
        - local_idx (int): Index for the current local time point.
        - year_idx (int): Index for the year in the self.years array.
        - time_step_hours (int, optional): Time step size in hours. Default is 6.

        Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tensors for input and target cosine zenith angles.
        """
        if not 0 <= year_idx < len(self.years):
            raise ValueError("year_idx is out of bounds.")
        
        year = self.years[year_idx]
        jan_01_epoch = datetime.datetime(year, 1, 1, 0, 0, 0)  # Reference datetime for the start of the year
           
        # Helper function to calculate cosine zenith angles
        def calculate_cos_zenith(start_idx: int, end_idx: int) -> np.ndarray:
            cos_zenith = []  # to store cosine zenith angles
            for idx in range(start_idx, end_idx, self.dt):
                hours_since_jan_01 = idx * time_step_hours  
                model_time = jan_01_epoch + datetime.timedelta(hours=hours_since_jan_01)  
                # Calculate and append the cosine of zenith angle for this time
                cos_zenith.append(
                    cos_zenith_angle(
                        model_time, self.lon_grid_local, self.lat_grid_local
                    ).astype(np.float32)
                )
            return np.stack(cos_zenith, axis=0)  # Stack the angles into a multi-dimensional array
        
        # Calculate the cosine zenith angles for the input and target time points
        cos_zenith_inp = calculate_cos_zenith(local_idx, local_idx + 1)
        cos_zenith_tar = calculate_cos_zenith(local_idx + self.dt, local_idx + self.dt * (self.n_future + 1) + 1)

        # Return the input and target angles as PyTorch tensors
        return torch.as_tensor(cos_zenith_inp), torch.as_tensor(cos_zenith_tar)

    def __getitem__(self, global_idx):
        year_idx = int(global_idx / self.n_samples_per_year)  # which year
        local_idx = int(global_idx % self.n_samples_per_year) # which sample in that year 

        # open image file
        if self.files[year_idx] is None:
            self._open_file(year_idx)
        step = self.dt # time step

        # boundary conditions to ensure we don't pull data that is not in a specific year
        local_idx = local_idx % (self.n_samples_per_year - step * (self.n_future + 1))
        if local_idx < step:
            local_idx += step
        
        # pre-process and get the image fields
        inp_field = self.files[year_idx][local_idx,self.in_channels,0:self.img_shape_x,0:self.img_shape_y]
        tar_field = self.files[year_idx][(local_idx + step):(local_idx + step * (self.n_future + 1) + 1):step, \
                                         self.out_channels,0:self.img_shape_x,0:self.img_shape_y]
        # flatten time indices
        tar_field = tar_field.reshape((self.n_out_channels * (self.n_future + 1), self.img_shape_x, self.img_shape_y))
        # normalize images if needed
        inp, tar = self._normalize(inp_field), self._normalize(tar_field)

        if self.params.add_zenith:
            zen_inp, zen_tar = self._compute_zenith_angle(local_idx, year_idx) #compute the zenith angles for the input.
            zen_inp = zen_inp[:,:self.img_shape_x] #adjust to match input dimensions
            zen_tar = zen_tar[:,:self.img_shape_x] #adjust to match input dimensions
            result = inp, tar, zen_inp, zen_tar
        else:
            result = inp, tar

        return result
