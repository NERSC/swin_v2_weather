import os
import glob
import numpy as np
import cupyx as cpx
import h5py
import logging
from modulus.utils.sfno.zenith_angle import cos_zenith_angle
from bisect import bisect_right
import operator
from itertools import accumulate, groupby
import datetime

class ERA5ES(object):
    # very important: the seed has to be constant across the workers, or otherwise mayhem:
    def __init__(self, location, 
                train, batch_size, 
                dt, img_size,
                n_in_channels, n_out_channels, 
                num_shards,
                shard_id,
                n_future,
                enable_logging=True,
                add_zenith=True,
                seed=333):
        self.batch_size = batch_size
        self.location = location
        self.img_size = img_size
        self.train = train
        self.dt = dt
        self.n_in_channels = n_in_channels
        self.n_out_channels = n_out_channels
        self.base_seed = seed
        self.num_shards = num_shards
        self.shard_id = shard_id
        self.add_zenith = add_zenith
        self.n_future = n_future
        if self.add_zenith:
            # additional static fields needed for coszen
            longitude = np.arange(0, 360, 0.25)
            latitude = np.arange(-90, 90.25, 0.25)
            latitude = latitude[::-1]
            self.lon_grid_local, self.lat_grid_local = np.meshgrid(longitude, latitude)
        
        self._get_files_stats(enable_logging)
        self.shuffle = True if train else False
        
    def _get_files_stats(self, enable_logging):
        self.files_paths = glob.glob(self.location + "/*.h5")
        self.files_paths.sort()
        self.years = [int(os.path.splitext(os.path.basename(x))[0][-4:]) for x in self.files_paths]
        self.n_years = len(self.files_paths)
        self.n_samples_year = []

        with h5py.File(self.files_paths[0], 'r') as _f:
            logging.info("Getting file stats from {}".format(self.files_paths[0]))
            self.n_samples_year.append(_f["fields"].shape[0])
            self.img_shape_x = self.img_size[0]
            self.img_shape_y = self.img_size[1]
            assert(self.img_shape_x <= _f['fields'].shape[2] and self.img_shape_y <= _f['fields'].shape[3]), 'image shapes are greater than dataset image shapes'

        self.year_offsets = list(accumulate(self.n_samples_year, operator.add))[:-1]
        self.year_offsets.insert(0, 0)
        self.n_samples_available = sum(self.n_samples_year)
        self.n_samples_total = self.n_samples_available
        self.n_samples_shard = self.n_samples_total // self.num_shards

        # number of steps per epoch
        self.num_steps_per_cycle = self.n_samples_shard // self.batch_size
        self.n_samples_per_epoch = self.n_samples_total # set to total; in modulus this can be smaller but not used currently
        self.num_steps_per_epoch = self.n_samples_per_epoch // (
            self.batch_size * self.num_shards
        )

        # we need those here
        self.num_samples_per_cycle_shard = self.num_steps_per_cycle * self.batch_size
        self.num_samples_per_epoch_shard = self.num_steps_per_epoch * self.batch_size

        self.files = [None for _ in range(self.n_years)]
        self.dsets = [None for _ in range(self.n_years)]
        if enable_logging:
            logging.info("Total samples: {}".format(self.n_samples_available))
            logging.info("Found data at path {}. Number of examples: {}. Image Shape: {} x {} x {}".format(self.location, self.n_samples_total, self.img_shape_x, self.img_shape_y, self.n_in_channels))
            if self.num_shards > 1:
                logging.info("Using shards of size {} per rank".format(self.n_samples_shard))

        # some state variables
        self.last_cycle_epoch = None
        self.index_permutation = None

        # prepare buffers for double buffering
        self.current_buffer = 0
        self.inp_buffs = [cpx.zeros_pinned((self.n_in_channels, self.img_shape_x, self.img_shape_y), dtype=np.float32),
                        cpx.zeros_pinned((self.n_in_channels, self.img_shape_x, self.img_shape_y), dtype=np.float32)]
        self.tar_buffs = [cpx.zeros_pinned((self.n_out_channels, self.img_shape_x, self.img_shape_y), dtype=np.float32),
                        cpx.zeros_pinned((self.n_out_channels, self.img_shape_x, self.img_shape_y), dtype=np.float32)]    
        if self.add_zenith:
            self.zen_inp_buffs = [cpx.zeros_pinned((1, self.img_shape_x, self.img_shape_y), dtype=np.float32),
                            cpx.zeros_pinned((1, self.img_shape_x, self.img_shape_y), dtype=np.float32)]
            self.zen_tar_buffs = [cpx.zeros_pinned((1, self.img_shape_x, self.img_shape_y), dtype=np.float32),
                            cpx.zeros_pinned((1, self.img_shape_x, self.img_shape_y), dtype=np.float32)]    

    
    def __len__(self):
        return self.n_samples_shard

    def __del__(self):
        for f in self.files:
            if f is not None:
                f.close()

    def _compute_zenith_angle(self, local_idx, year_idx, time_step_hours=6):
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
        return (cos_zenith_inp), (cos_zenith_tar)

    def __call__(self, sample_info):
        # compute global iteration index:
        global_sample_idx = sample_info.idx_in_epoch + sample_info.epoch_idx * self.num_samples_per_epoch_shard
        cycle_sample_idx = global_sample_idx % self.num_samples_per_cycle_shard
        cycle_epoch_idx = global_sample_idx // self.num_samples_per_cycle_shard

        # check if epoch is done
        if sample_info.iteration >= self.num_steps_per_epoch:
            raise StopIteration

        # shuffle the data and shard
        if cycle_epoch_idx != self.last_cycle_epoch:
            self.last_cycle_epoch = cycle_epoch_idx
            # generate a unique seed and permutation:
            rng = np.random.default_rng(seed=self.base_seed + cycle_epoch_idx)
            if self.shuffle:
                self.index_permutation = rng.permutation(self.n_samples_total)
            else:
                self.index_permutation = np.arange(self.n_samples_total)
            # shard the data
            start = self.n_samples_shard * self.shard_id
            end = start + self.n_samples_shard
            self.index_permutation = self.index_permutation[start:end]

        # determine local and sample idx
        sample_idx = self.index_permutation[cycle_sample_idx]
        year_idx = bisect_right(self.year_offsets, sample_idx) - 1 # subtract 1 because we do 0-based indexing
        local_idx = sample_idx - self.year_offsets[year_idx]

        step = self.dt # time step

        # boundary conditions to ensure we don't pull data that is not in a specific year
        if local_idx < step:
            local_idx += step

        if local_idx >= (self.n_samples_year[year_idx] - step * (self.n_future + 1)):
            local_idx = self.n_samples_year[year_idx] - step * (self.n_future + 1) - 1
 
        if self.files[year_idx] is None:
            self.files[year_idx] = h5py.File(self.files_paths[year_idx], 'r')
            self.dsets[year_idx] = self.files[year_idx]['fields']
        
        tmp_inp = self.dsets[year_idx][local_idx, ...]
        tmp_tar = self.dsets[year_idx][local_idx+step, ...]

        # handles to buffers buffers
        inp = self.inp_buffs[self.current_buffer]
        tar = self.tar_buffs[self.current_buffer]
        if self.add_zenith:
            zen_inp = self.zen_inp_buffs[self.current_buffer]
            zen_tar = self.zen_tar_buffs[self.current_buffer]
        self.current_buffer = (self.current_buffer + 1) % 2
        
        # crop the pixels:
        inp[...] = tmp_inp[..., :self.img_shape_x, :self.img_shape_y]
        tar[...] = tmp_tar[..., :self.img_shape_x, :self.img_shape_y]

        if self.add_zenith:
            tmp_zen_inp, tmp_zen_tar = self._compute_zenith_angle(local_idx, year_idx) #compute the zenith angles for the input.
            zen_inp[...] = tmp_zen_inp[...,:self.img_shape_x,:self.img_shape_y] #adjust to match input dimensions
            zen_tar[...] = tmp_zen_tar[...,:self.img_shape_x,:self.img_shape_y] #adjust to match input dimensions
            result = inp, tar, zen_inp, zen_tar
        else:
            result = inp, tar

        return result
