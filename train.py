import os
import sys
import time
import numpy as np
import argparse
import h5py
import torch
import wandb
import matplotlib.pyplot as plt
from collections import OrderedDict
from typing import Callable, Any

# opt
import torch.cuda.amp as amp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from apex import optimizers

# logging, yparams
import logging
from utils import logging_utils
logging_utils.config_logger()
from utils.YParams import YParams
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap as ruamelDict

# metrics, utils, data
from utils.data_loader_era5 import get_data_loader
from utils.weighted_acc_rmse import weighted_rmse_torch
from utils.loss import LpLoss, GeometricLpLoss
from utils.img_utils import vis

from networks.helpers import get_model

def ckpt_identity(layer: Callable, *args: Any, **kwargs: Any) -> Any:
    """Identity function for when activation checkpointing is not needed"""
    return layer(*args)

def set_seed(params, world_size):
    seed = params.seed
    if seed is None:
        seed = np.random.randint(10000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if world_size > 0:
        torch.cuda.manual_seed_all(seed)

class Trainer():
    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def __init__(self, params, args):
        self.sweep_id = args.sweep_id
        self.root_dir = params['exp_dir'] 
        self.config = args.config
        params['enable_amp'] = args.enable_amp

        self.world_size = 1
        if 'WORLD_SIZE' in os.environ:
            self.world_size = int(os.environ['WORLD_SIZE'])

        self.local_rank = 0
        self.world_rank = 0
        if self.world_size > 1:
            dist.init_process_group(backend='nccl',
                                    init_method='env://')
            self.world_rank = dist.get_rank()
            self.local_rank = int(os.environ["LOCAL_RANK"])

        torch.cuda.set_device(self.local_rank)
        torch.backends.cudnn.benchmark = True
        
        self.log_to_screen = params.log_to_screen and self.world_rank==0
        self.log_to_wandb = params.log_to_wandb and self.world_rank==0
    
        self.device = torch.cuda.current_device()
        self.params = params
        self.params['name'] = args.config + '_' + str(args.run_num)
        self.params['group'] = args.config
        self.config = args.config 
        self.run_num = args.run_num
        self.ckpt_fn = torch.utils.checkpoint.checkpoint if hasattr(params, 'activation_ckpt') and params.activation_ckpt else ckpt_identity
    
    def build_and_launch(self):
        self.params['in_channels'] = np.array(self.params['in_channels'])
        self.params['out_channels'] = np.array(self.params['out_channels'])
        self.params['n_in_channels'] = len(self.params['in_channels'])
        self.params['n_out_channels'] = len(self.params['out_channels'])

        if self.params.add_zenith:
            self.params.n_in_channels += 1

        # init wandb
        if self.sweep_id:
            jid = os.environ['SLURM_JOBID'] # so different sweeps dont resume
            exp_dir = os.path.join(*[self.root_dir, 'sweeps', self.sweep_id, self.config, jid])
        else:
            exp_dir = os.path.join(*[self.root_dir, self.config, self.run_num])

        if self.world_rank==0:
            if not os.path.isdir(exp_dir):
                os.makedirs(exp_dir)
                os.makedirs(os.path.join(exp_dir, 'training_checkpoints/'))
                os.makedirs(os.path.join(exp_dir, 'wandb/'))

        self.params['experiment_dir'] = os.path.abspath(exp_dir)
        self.params['checkpoint_path'] = os.path.join(exp_dir, 'training_checkpoints/ckpt.tar')
        self.params['best_checkpoint_path'] = os.path.join(exp_dir, 'training_checkpoints/best_ckpt.tar')
        self.params['resuming'] = True if os.path.isfile(self.params.checkpoint_path) else False
        if self.log_to_wandb:
            if self.sweep_id:
                wandb.init(dir=os.path.join(exp_dir, "wandb"))
                hpo_config = wandb.config
                self.params.update_params(hpo_config)
                logging.info('HPO sweep %s, trial params:'%self.sweep_id)
                logging.info(self.params.log())
            else:
                wandb.init(dir=os.path.join(exp_dir, "wandb"),
                            config=self.params.params, name=self.params.name, group=self.params.group, project=self.params.project, 
                            entity=self.params.entity, resume=self.params.resuming)
                logging.info(self.params.log())

        if self.sweep_id and dist.is_initialized():
            # broadcast the params to all ranks since the sweep agent has changed it
            if self.world_rank == 0: # where the wandb agent has changed params
                objects = [self.params]
            else:
                self.params = None
                objects = [None]

            dist.broadcast_object_list(objects, src=0)
            self.params = objects[0]

        # set_seed(self.params, self.world_size)

        if self.world_rank==0:
            logging_utils.log_to_file(logger_name=None, log_filename=os.path.join(exp_dir, 'out.log'))
            logging_utils.log_versions()

        self.params['global_batch_size'] = self.params.batch_size
        self.params['local_batch_size'] = int(self.params.batch_size//self.world_size)

        self.train_data_loader, self.train_dataset, self.train_sampler = get_data_loader(self.params, self.params.train_data_path, dist.is_initialized(), train=True)
        self.valid_data_loader, self.valid_dataset = get_data_loader(self.params, self.params.valid_data_path, dist.is_initialized(), train=False)

        self.params['img_shape_x'] = self.train_dataset.img_shape_x
        self.params['img_shape_y'] = self.train_dataset.img_shape_y

        # dump the yaml used
        if self.world_rank == 0:
            hparams = ruamelDict()
            yaml = YAML()
            for key, value in self.params.params.items():
                hparams[str(key)] = value.tolist() if isinstance(value, np.ndarray) else value
                with open(os.path.join(self.params['experiment_dir'], 'hyperparams.yaml'), 'w') as hpfile:
                    yaml.dump(hparams, hpfile)

        if self.params.loss_type == 'l2':
            self.loss_obj = LpLoss()
        elif self.params.loss_type == 'geo':
            self.loss_obj = GeometricLpLoss(img_size=tuple(self.params.img_size), device=self.device)
        self.model = get_model(self.params).to(self.device) 

        if self.log_to_wandb:
            wandb.watch(self.model)

        if self.params.optimizer_type == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr =self.params.lr, betas=(0.9, 0.95), fused=True)
        elif self.params.optimizer_type == 'FusedLAMB':
            self.optimizer = optimizers.FusedLAMB(self.model.parameters(), lr = self.params.lr, max_grad_norm=5.)
        else:
            raise Exception(f"optimizer type {self.params.optimizer_type} not implemented")


        if self.params.enable_amp == True:
            self.gscaler = amp.GradScaler()

        if dist.is_initialized():
            self.model = DistributedDataParallel(self.model,
                                                device_ids=[self.local_rank],
                                                output_device=[self.local_rank], 
                                                static_graph=(params.checkpointing>0))

        self.iters = 0
        self.startEpoch = 0

        if self.params.finetune and not self.params.resuming:
            assert (
                params.pretrained_checkpoint_path is not None
            ), "error, please specify a valid pretrained checkpoint path"
            if self.log_to_screen:
                logging.info("Loading checkpoint %s"%self.params.pretrained_checkpoint_path)
            self.restore_checkpoint(params.pretrained_checkpoint_path)

        if self.params.resuming:
            if self.log_to_screen:
                logging.info("Loading checkpoint %s"%self.params.checkpoint_path)
            self.restore_checkpoint(self.params.checkpoint_path)
                
        self.epoch = self.startEpoch

        if self.params.scheduler == 'ReduceLROnPlateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.2, patience=5, mode='min')
        elif self.params.scheduler == 'CosineAnnealingLR':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.params.max_epochs, last_epoch=self.startEpoch-1)
        else:
            self.scheduler = None

        if self.log_to_screen:
            logging.info(self.model)

        # launch training
        self.train()

    def train(self):
        if self.log_to_screen:
            logging.info("Starting Training Loop...")

        best_valid_loss = 1.e6
        for epoch in range(self.startEpoch, self.params.max_epochs):
            if dist.is_initialized():
                self.train_sampler.set_epoch(epoch)

            start = time.time()

            tr_time, data_time, train_logs = self.train_one_epoch()
            valid_time, valid_logs = self.validate_one_epoch()

            if self.params.scheduler == 'ReduceLROnPlateau':
                self.scheduler.step(valid_logs['valid_loss'])
            elif self.params.scheduler == 'CosineAnnealingLR':
                self.scheduler.step()

            if self.log_to_wandb:
                for pg in self.optimizer.param_groups:
                    lr = pg['lr']
                wandb.log({'lr': lr})

            if self.world_rank == 0:
                if self.params.save_checkpoint:
                    #checkpoint at the end of every epoch
                    self.save_checkpoint(self.params.checkpoint_path)
                    if valid_logs['valid_loss'] <= best_valid_loss:
                        #logging.info('Val loss improved from {} to {}'.format(best_valid_loss, valid_logs['valid_loss']))
                        self.save_checkpoint(self.params.best_checkpoint_path)
                        best_valid_loss = valid_logs['valid_loss']

            if self.log_to_screen:
                logging.info('Time taken for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
                logging.info('Train loss: {}. Valid loss: {}'.format(train_logs['loss'], valid_logs['valid_loss']))


    def train_one_epoch(self):
        self.epoch += 1
        tr_time = 0
        data_time = 0
        tr_loss = []
        self.model.train()
        
        for i, data in enumerate(self.train_data_loader, 0):
            tr_start = time.time()
            inp, tar = map(lambda x: x.to(self.device, dtype = torch.float), data)      
            self.model.zero_grad()
            with amp.autocast(self.params.enable_amp):
                gen = self.model(inp).to(self.device, dtype = torch.float)
                loss = self.loss_obj(gen, tar)

            if self.params.enable_amp:
                self.gscaler.scale(loss).backward()
                self.gscaler.step(self.optimizer)
            else:
                loss.backward()
                self.optimizer.step()

            if self.params.enable_amp:
                self.gscaler.update()

            # logs
            if dist.is_initialized():
                dist.all_reduce(loss)
            tr_loss.append(loss.item()/dist.get_world_size())

            tr_time += time.time() - tr_start
        
        logs = {'loss': np.mean(tr_loss)}

        if self.log_to_wandb:
            wandb.log(logs, step=self.epoch)

        return tr_time, data_time, logs

    def validate_one_epoch(self):
        self.model.eval()

        mult = torch.as_tensor(np.load(self.params.global_stds_path)[0,self.params.out_channels,0,0]).to(self.device)
        valid_buff = torch.zeros((3), dtype=torch.float32, device=self.device)
        valid_loss = valid_buff[0].view(-1)
        valid_steps = valid_buff[2].view(-1)
        valid_weighted_rmse = torch.zeros((self.params.n_out_channels), dtype=torch.float32, device=self.device)
        valid_weighted_acc = torch.zeros((self.params.n_out_channels), dtype=torch.float32, device=self.device)

        valid_start = time.time()

        sample_idx = np.random.randint(len(self.valid_data_loader))
        with torch.no_grad():
            for i, data in enumerate(self.valid_data_loader, 0):
                inp, tar  = map(lambda x: x.to(self.device, dtype = torch.float), data)
                gen = self.model(inp).to(self.device, dtype = torch.float)
                valid_loss += self.loss_obj(gen, tar) 
                valid_steps += 1.
                valid_weighted_rmse += weighted_rmse_torch(gen, tar)

                if (i == sample_idx) and self.log_to_wandb:
                    fields = [gen[0,0].detach().cpu().numpy(), tar[0,0].detach().cpu().numpy()]

            
        if dist.is_initialized():
            dist.all_reduce(valid_buff)
            dist.all_reduce(valid_weighted_rmse)

        # divide by number of steps
        valid_buff[0:2] = valid_buff[0:2] / valid_buff[2]
        valid_weighted_rmse = valid_weighted_rmse / valid_buff[2]
        valid_weighted_rmse *= mult

        # download buffers
        valid_buff_cpu = valid_buff.detach().cpu().numpy()
        valid_weighted_rmse_cpu = valid_weighted_rmse.detach().cpu().numpy()

        valid_time = time.time() - valid_start
        valid_weighted_rmse = mult*torch.mean(valid_weighted_rmse, axis = 0)
        logs = {'valid_loss': valid_buff_cpu[0]}

        # track specific variables
        if hasattr(self.params, 'track_channels'):
            idxes = [self.params.channel_names.index(varname) for varname in self.params.track_channels]
            track_channels = self.params.track_channels
        else:
            track_channels = ['u10m', 'v10m']
            idxes = [0, 1]

        for idx,var in zip(idxes,track_channels):
            logs.update({f'valid_rmse_{var}': valid_weighted_rmse_cpu[idx]})
        
        if self.log_to_wandb:
            fig = vis(fields)
            logs['vis'] = wandb.Image(fig)
            plt.close(fig)
            wandb.log(logs, step=self.epoch)

        return valid_time, logs


    def save_checkpoint(self, checkpoint_path, model=None):
        if not model:
            model = self.model
        torch.save({'iters': self.iters, 'epoch': self.epoch, 'model_state': model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()}, checkpoint_path)

    def restore_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cuda:{}'.format(self.local_rank))
        try:
            self.model.load_state_dict(checkpoint['model_state'])
        except:
            new_state_dict = OrderedDict()
            for key, val in checkpoint['model_state'].items():
                name = key[7:]
                new_state_dict[name] = val 
            self.model.load_state_dict(new_state_dict)
        if self.params.resuming:
            self.iters = checkpoint['iters']
            self.startEpoch = checkpoint['epoch']
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_num", default='00', type=str)
    parser.add_argument("--yaml_config", default='./config/afno.yaml', type=str)
    parser.add_argument("--config", default='default', type=str)
    parser.add_argument("--enable_amp", action='store_true')
    parser.add_argument("--sweep_id", default=None, type=str, help='sweep config from ./configs/sweeps.yaml')
    args = parser.parse_args()

    params = YParams(os.path.abspath(args.yaml_config), args.config)
    trainer = Trainer(params, args)

    if args.sweep_id and trainer.world_rank==0:
        wandb.agent(args.sweep_id, function=trainer.build_and_launch, count=1, entity=trainer.params.entity, project=trainer.params.project)
    else:
        trainer.build_and_launch()

    if dist.is_initialized():
        dist.barrier()

    logging.info('DONE')
