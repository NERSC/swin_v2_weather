import logging
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch import Tensor
import types

#concurrent futures
import concurrent.futures as cf

# distributed stuff
import torch.distributed as dist

#dali stuff
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

# es helper
import utils.dali_era5_es_helper as esh

def get_data_loader(params, files_pattern, distributed, train):
    dataloader = DaliDataLoader(params, files_pattern, train)

    dataset = types.SimpleNamespace(img_shape_x=dataloader.img_shape_x,
                                    img_shape_y=dataloader.img_shape_y)

    if train:
        return dataloader, dataset, None
    else:
        return dataloader, dataset

class DaliDataLoader(object):
    def get_pipeline(self):
        pipeline = Pipeline(batch_size = self.batch_size,
                            num_threads = 2,
                            device_id = self.device_index,
                            py_num_workers = self.num_data_workers,
                            py_start_method='spawn',
                            seed = self.global_seed)
        
     
        with pipeline: # get input and target 
            # get input and target
            data = fn.external_source(source = esh.ERA5ES(self.location,
                                                          self.train,
                                                          self.batch_size,
                                                          self.dt,
                                                          self.img_size,
                                                          self.n_in_channels,
                                                          self.n_out_channels,
                                                          self.num_shards,
                                                          self.shard_id,
                                                          self.n_future,
                                                          enable_logging = False,
                                                          add_zenith=self.add_zenith,
                                                          seed=self.global_seed),
                                          num_outputs = 4 if self.add_zenith else 2,
                                          layout = ["CHW", "CHW"],
                                          batch = False,
                                          no_copy = True,
                                          parallel = True,
                                          prefetch_queue_depth = self.num_data_workers)
            
            if self.add_zenith:
                inp, tar, izen, tzen = data
            else:
                inp, tar = data

            # upload to GPU
            inp = inp.gpu()
            tar = tar.gpu()
            if self.add_zenith:
                izen = izen.gpu()
                tzen = tzen.gpu()

            if self.normalize:
                inp = fn.normalize(inp,
                                   device = "gpu",
                                   axis_names = "HW",
                                   batch = False,
                                   mean = self.in_bias,
                                   stddev = self.in_scale)

                tar = fn.normalize(tar,
                                   device = "gpu",
                                   axis_names = "HW",
                                   batch = False,
                                   mean = self.out_bias,
                                   stddev = self.out_scale)

            # add zenith angle if requested
            if self.add_zenith:
                pipeline.set_outputs(inp, tar, izen, tzen)
            else:
                pipeline.set_outputs(inp, tar)

        return pipeline

    def __init__(self, params, location, train, seed = 333):
        # set up seeds
        # this one is the same on all ranks
        self.global_seed = seed
        # this seed is supposed to be diffferent for every rank
        self.local_seed = self.global_seed + dist.get_rank()

        self.num_data_workers = params.num_data_workers
        self.device_index = torch.cuda.current_device()
        self.batch_size = int(params.local_batch_size)

        self.location = location
        self.train = train
        self.dt = params.dt
        self.in_channels = params.in_channels
        self.out_channels = params.out_channels
        self.n_in_channels = len(self.in_channels)
        self.n_out_channels = len(self.out_channels)
        self.n_future = params.n_future
        self.img_size = params.img_size
        self.add_zenith = params.add_zenith

        # load stats
        self.normalize = True
        means = np.load(params.global_means_path)[0][:self.n_in_channels]
        stds = np.load(params.global_stds_path)[0][:self.n_in_channels]
        self.in_bias = means
        self.in_scale = stds
        means = np.load(params.global_means_path)[0][:self.n_out_channels]
        stds = np.load(params.global_stds_path)[0][:self.n_out_channels]
        self.out_bias = means
        self.out_scale = stds

        # set sharding
        if dist.is_initialized():
            self.num_shards = params.data_num_shards
            self.shard_id = params.data_shard_id
        else:
            self.num_shards = 1
            self.shard_id = 0

        # get img source data
        extsource = esh.ERA5ES(self.location,
                               self.train,
                               self.batch_size,
                               self.dt,
                               self.img_size,
                               self.n_in_channels,
                               self.n_out_channels,
                               self.num_shards,
                               self.shard_id,
                               self.n_future,
                               add_zenith=self.add_zenith,
                               seed=self.global_seed)
        self.num_batches = extsource.num_steps_per_epoch
        self.img_shape_x = extsource.img_shape_x
        self.img_shape_y = extsource.img_shape_y
        del extsource
 
         # create pipeline
        self.pipeline = self.get_pipeline()
        self.pipeline.start_py_workers()
        self.pipeline.build()

        # create iterator
        outnames = ["inp", "tar"]
        if self.add_zenith:
            outnames += ["izen", "tzen"]

        # create iterator
        self.iterator = DALIGenericIterator(
            [self.pipeline],
            outnames,
            auto_reset=True,
            size=-1,
            last_batch_policy=LastBatchPolicy.DROP,
            prepare_first_batch=True,
        )
        
    def __len__(self):
        return self.num_batches

    def __iter__(self):
        #self.iterator.reset()
        for token in self.iterator:
            inp = token[0]["inp"]
            tar = token[0]["tar"]

            if self.add_zenith:
                izen = token[0]["izen"]
                tzen = token[0]["tzen"]
                result = inp, tar, izen, tzen
            else:
                result = inp, tar

            yield result
