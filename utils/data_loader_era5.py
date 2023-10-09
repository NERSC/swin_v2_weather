import os
import logging
import glob
import torch
import random
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch import Tensor
import h5py

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

        return inp, tar
