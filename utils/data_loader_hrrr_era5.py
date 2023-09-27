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
from datetime import datetime, timedelta
import calendar
import zarr
import xarray as xr


def worker_init(wrk_id):
    np.random.seed(torch.utils.data.get_worker_info().seed%(2**32 - 1))

def get_data_loader(params, distributed, train):
    dataset = HrrrEra5Dataset(params, train)
    sampler = DistributedSampler(dataset, shuffle=train) if distributed else None
    
    dataloader = DataLoader(dataset,
                            batch_size=int(params.local_batch_size),
                            num_workers=params.num_data_workers,
                            shuffle=(sampler is None),
                            sampler=sampler if train else None,
                            worker_init_fn=worker_init,
                            drop_last=True,
                            pin_memory=torch.cuda.is_available())

    if train:
        return dataloader, dataset, sampler
    else:
        return dataloader, dataset



class HrrrEra5Dataset(Dataset):
    '''
    Paired dataset object serving time-synchronized pairs of ERA5 and HRRR samples
    Expects data to be stored under directory specified by 'location'
        ERA5 under <root_dir>/era5/
        HRRR under <root_dir>/hrrr/
    '''
    def __init__(self, params, train):
        self.params = params
        self.location = params.location
        self.train = train
        self.path_suffix = 'train' if train else 'valid'
        self.dt = params.dt
        self.normalize = False #TODO implement HRRR+ERA5 normalization
        self._get_files_stats()
        #self.means_hrrr = np.load(os.path.join(self.location, 'hrrr', 'stats', 'means.npy'))
        #self.stds_hrrr = np.load(os.path.join(self.location, 'hrrr', 'stats', 'stds.npy'))
        

    def _get_files_stats(self):
        '''
        Scan directories and extract metadata for ERA5 and HRRR
        '''

        # ERA5 parsing
        self.era5_paths = glob.glob(os.path.join(self.location,"era5", self.path_suffix, "*.zarr"))
        self.era5_paths.sort()
        self.years = [int(os.path.basename(x).replace('.zarr', '')) for x in self.era5_paths]
        self.n_years = len(self.era5_paths)

        with xr.open_zarr(self.era5_paths[0], consolidated=True) as ds:
            self.era5_channels = ds.channel
            self.era5_lat = ds.latitude
            self.era5_lon = ds.longitude

        self.n_samples_total = self.compute_total_samples()
        self.ds_era5 = [None for _ in range(self.n_years)]

        # HRRR parsing
        self.hrrr_paths = glob.glob(os.path.join(self.location, "hrrr", self.path_suffix, "*.zarr"))
        self.hrrr_paths.sort()
        years = [int(os.path.basename(x).replace('.zarr', '')) for x in self.hrrr_paths]
        assert years == self.years, 'Number of years for ERA5 in %s and HRRR in %s must match'%(os.path.join(self.location, "era5/*.zarr"),
                                                                                                os.path.join(self.location, "hrrr/*.zarr"))
        with xr.open_zarr(self.hrrr_paths[0], consolidated=True) as ds:
            self.hrrr_channels = ds.channel
            self.hrrr_lat = ds.latitude.isel(time=0) # TODO remove the .isel(time=0) once the time dim is removed from hrrr zarr lat/lon
            self.hrrr_lon = ds.longitude.isel(time=0)
            self.era5_lat_window, self.era5_lon_window = self._construct_era5_window()
        self.ds_hrrr = [None for _ in range(self.n_years)]
    
    def __len__(self):
        return self.n_samples_total


    def compute_total_samples(self):
        '''
        Loop through all years and count the total number of samples
        '''
        print(self.years)
        first_year = sorted(self.years)[0]
        last_year = sorted(self.years)[-1]
        first_sample = datetime(year=first_year, month=1, day=1, hour=1, minute=0, second=0)
        last_sample = datetime(year=last_year, month=12, day=31, hour=23, minute=0, second=0)

        all_datetimes = [first_sample + timedelta(hours=x) for x in range(int((last_sample-first_sample).total_seconds()/3600)+1)]

        missing_samples = np.load(os.path.join(self.location, 'missing_samples.npy'), allow_pickle=True)

        self.valid_samples = [x for x in all_datetimes if (x not in missing_samples) and (x + timedelta(hours=self.dt) not in missing_samples)]

        logging.info('Total datetimes in training set are {} of which {} are valid'.format(len(all_datetimes), len(self.valid_samples)))
        print(len(self.valid_samples))

        return len(self.valid_samples)

    def _normalize_era5(self, img):
        if self.normalize:
            img -= self.means_era5
            img /= self.stds_era5
        return torch.as_tensor(img)

    def _get_era5(self, ts_inp, ts_tar):
        '''
        Retrieve ERA5 samples from zarr files
        '''
        ds_inp, ds_tar, adjacent = self._get_ds_handles(self.ds_era5, self.era5_paths, ts_inp, ts_tar)

        # TODO update to use a fixed boundary beyond the HRRR domain, determined by e.g. self.era5_window_lat, self.era5_window_lon = self._construct_era5_window()
        # If we use a lambert projection like the interp example below, need to precompute those
        inp_field = ds_inp.sel(time=ts_inp, channel=self.era5_channels).interp(latitude=self.hrrr_lat, longitude=self.hrrr_lon).data.values
        tar_field = ds_inp.sel(time=ts_tar, channel=self.era5_channels).interp(latitude=self.hrrr_lat, longitude=self.hrrr_lon).data.values
        
        inp, tar = self._normalize_era5(inp_field), self._normalize_era5(tar_field)

        return inp, tar

    def _normalize_hrrr(self, img):
        if self.normalize:
            img -= self.means_hrrr
            img /= self.stds_hrrr
        return torch.as_tensor(img)

    def _get_hrrr(self, ts_inp, ts_tar):
        '''
        Retrieve HRRR samples from zarr files
        '''
        ds_inp, ds_tar, adjacent = self._get_ds_handles(self.ds_hrrr, self.hrrr_paths, ts_inp, ts_tar)

        inp_field = ds_inp.sel(time=ts_inp).HRRR.values
        tar_field = ds_tar.sel(time=ts_tar).HRRR.values
        inp, tar = self._normalize_hrrr(inp_field), self._normalize_hrrr(tar_field)

        return inp, tar

    def __getitem__(self, global_idx):
        '''
        Return data as a dict (so we can potentially add extras, metadata, etc if desired
        '''
        time_pair = self._global_idx_to_datetime(global_idx)
        era5_pair = self._get_era5(*time_pair)
        hrrr_pair = self._get_hrrr(*time_pair)
        return {
                'era5':era5_pair,
                'hrrr':hrrr_pair,
                #'time':(np.datetime64(time_pair[0]), np.datetime64(time_pair[1]))
               }

    def _global_idx_to_datetime(self, global_idx):
        '''
        Parse a global sample index and return the input/target timstamps as datetimes
        '''
        #base = datetime(self.years[0],1,1,1,0) # since HRRR indexing for each year starts at 00:01:00 UTC
        #inp = base + timedelta(hours=global_idx)
        #tar = base + timedelta(hours=global_idx + self.dt)

        inp = self.valid_samples[global_idx]
        tar = inp + timedelta(hours=self.dt)

        return inp, tar

    def _get_ds_handles(self, handles, paths, ts_inp, ts_tar):
        '''
        Return opened dataset handles for the appropriate year, and boolean indicating if they are from the same year
        '''
        ds_handles = []
        for year in [ts_inp.year, ts_tar.year]:
            year_idx = self.years.index(year)
            if handles[year_idx] is None:
                handles[year_idx] = xr.open_zarr(paths[year_idx], consolidated=True)
            handles.append(handles[year_idx])
        return handles[0], handles[1], handles[0]==handles[1]

    def _construct_era5_window(self):
        '''
        Build custom indexing window to subselect HRRR region from ERA5
        '''
        # TODO implement a fixed lat/lon boundary outside of the HRRR region
        # Example below just uses the bounds of HRRR directly, but would index the equiangular lat-lon porjection of ERA5
        lat_lo_idx, lat_hi_idx = [np.argmin(np.abs(self.era5_lat.values - x)) for x in [self.hrrr_lat.values.min(), self.hrrr_lat.values.max()]] 
        lon_lo_idx, lon_hi_idx = [np.argmin(np.abs(self.era5_lon.values - x)) for x in [self.hrrr_lon.values.min(), self.hrrr_lon.values.max()]]
        return slice(lat_hi_idx, lat_lo_idx), slice(lon_lo_idx, lon_hi_idx)

