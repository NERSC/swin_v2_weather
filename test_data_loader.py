import torch
from utils.data_loader_multifiles import get_data_loader
import numpy as np
import random
import h5py
from utils.YParams import YParams
from networks.afnonet import AFNONet, PrecipNet
import matplotlib.pyplot as plt

config_name ='afno_backbone_26var' 
params = YParams('config/AFNO.yaml', config_name)

params.batch_size = 1
dataloader, dataset,  sampler = get_data_loader(params, params.train_data_path, distributed=False, train=True)
valid_dataloader , sampler_valid = get_data_loader(params, params.valid_data_path, distributed=False, train=False)
params.crop_size_x = sampler_valid.crop_size_x
params.crop_size_y = sampler_valid.crop_size_y
params.img_shape_x = sampler_valid.img_shape_x
params.img_shape_y = sampler_valid.img_shape_y
params['in_channels'] = np.array(params['in_channels'])
params['out_channels'] = np.array(params['out_channels'])
params['N_in_channels'] = len(params['in_channels'])
params['N_out_channels'] = len(params['out_channels'])


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
params.device = device
model = AFNONet(params).to(device) 

iters = 0
train = True
with torch.no_grad():
  for i, data in enumerate(valid_dataloader, 0):
    if i > 1:
        break
    iters += 1
    inp, tar = map(lambda x: x.to(device, dtype = torch.float), data)
    print(inp.shape)
    print(tar.shape)
    plt.rcParams["figure.figsize"] = (20,20)
    plt.figure()
    for ch in range(inp.shape[1]):
        plt.subplot(inp.shape[1],1, ch+1)
        plt.imshow(inp[0,ch,:,:].cpu(), cmap = 'RdBu')
        plt.colorbar()
    plt.savefig("pdfs/minibatch_" + str(i) + ".jpg")
    gen = model(inp)
    print(gen.shape)

