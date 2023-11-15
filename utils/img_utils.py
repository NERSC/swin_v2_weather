import h5py
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T


def interpolate(inp, tar, scale):
    sh = inp.shape
    transform = T.Resize((sh[1]//scale[0], sh[2]//scale[1]))
    return transform(inp), transform(tar)

def vis(fields):
    pred, tar = fields
    fig, ax = plt.subplots(1, 2, figsize=(24,12))
    ax[0].imshow(pred, cmap="turbo")
    ax[0].set_title("pred")
    ax[1].imshow(tar, cmap="turbo")
    ax[1].set_title("tar")
    fig.tight_layout()
    return fig

def reshape_fields(img, inp_or_tar, crop_size_x, crop_size_y,rnd_x, rnd_y, params, y_roll, train, normalize=True, orog=None, add_noise=False):
    #Takes in np array of size (n_history+1, c, h, w) and returns torch tensor of size ((n_channels*(n_history+1), crop_size_x, crop_size_y)

    if len(np.shape(img)) ==3:
      img = np.expand_dims(img, 0)

    sh = img.shape
    img = img[:, :, 0:(sh[2] - params.img_shape_x_remove_pixel)] #remove last pixel 

    n_history = np.shape(img)[0] - 1
    img_shape_x = np.shape(img)[-2]
    img_shape_y = np.shape(img)[-1]
    n_channels = np.shape(img)[1] #this will either be N_in_channels or N_out_channels
    channels = params.in_channels if inp_or_tar =='inp' else params.out_channels
    means = np.load(params.global_means_path)[:, channels]
    stds = np.load(params.global_stds_path)[:, channels]
    if crop_size_x == None:
        crop_size_x = img_shape_x
    if crop_size_y == None:
        crop_size_y = img_shape_y

    if normalize:
        if params.normalization == 'minmax':
          raise Exception("minmax not supported. Use zscore")
        elif params.normalization == 'zscore':
          img -=means
          img /=stds

    if params.add_grid:
        if inp_or_tar == 'inp':
            if params.gridtype == 'linear':
                assert params.N_grid_channels == 2, "N_grid_channels must be set to 2 for gridtype linear"
                x = np.meshgrid(np.linspace(-1, 1, img_shape_x))
                y = np.meshgrid(np.linspace(-1, 1, img_shape_y))
                grid_x, grid_y = np.meshgrid(y, x)
                grid = np.stack((grid_x, grid_y), axis = 0)
            elif params.gridtype == 'sinusoidal':
                assert params.N_grid_channels == 4, "N_grid_channels must be set to 4 for gridtype sinusoidal"
                x1 = np.meshgrid(np.sin(np.linspace(0, 2*np.pi, img_shape_x)))
                x2 = np.meshgrid(np.cos(np.linspace(0, 2*np.pi, img_shape_x)))
                y1 = np.meshgrid(np.sin(np.linspace(0, 2*np.pi, img_shape_y)))
                y2 = np.meshgrid(np.cos(np.linspace(0, 2*np.pi, img_shape_y)))
                grid_x1, grid_y1 = np.meshgrid(y1, x1)
                grid_x2, grid_y2 = np.meshgrid(y2, x2)
                grid = np.expand_dims(np.stack((grid_x1, grid_y1, grid_x2, grid_y2), axis = 0), axis = 0)
            img = np.concatenate((img, grid), axis = 1 )

    if params.orography and inp_or_tar == 'inp':
        img = np.concatenate((img, np.expand_dims(orog, axis = (0,1) )), axis = 1)
        n_channels += 1

    if params.roll:
        img = np.roll(img, y_roll, axis = -1)

    if train and (crop_size_x or crop_size_y):
        img = img[:,:,rnd_x:rnd_x+crop_size_x, rnd_y:rnd_y+crop_size_y]

    if inp_or_tar == 'inp':
        img = np.reshape(img, (n_channels*(n_history+1), crop_size_x, crop_size_y))
    elif inp_or_tar == 'tar':
        if params.two_step_training:
            img = np.reshape(img, (n_channels*2, crop_size_x, crop_size_y))
        else:
            img = np.reshape(img, (n_channels, crop_size_x, crop_size_y))

    if add_noise:
        img = img + np.random.normal(0, scale=params.noise_std, size=img.shape)

    return torch.as_tensor(img)
         


def reshape_precip(img, inp_or_tar, crop_size_x, crop_size_y,rnd_x, rnd_y, params, y_roll, train, normalize=True):

    if len(np.shape(img)) ==2:
      img = np.expand_dims(img, 0)

    sh = img.shape
    img = img[:, :, 0:(sh[2] - params.img_shape_x_remove_pixel)] #remove last pixel 
    img_shape_x = img.shape[-2]
    img_shape_y = img.shape[-1]
    n_channels = 1
    if crop_size_x == None:
        crop_size_x = img_shape_x
    if crop_size_y == None:
        crop_size_y = img_shape_y

    if normalize:
        eps = params.precip_eps
        img = np.log1p(img/eps)
    if params.add_grid:
        if inp_or_tar == 'inp':
            if params.gridtype == 'linear':
                assert params.N_grid_channels == 2, "N_grid_channels must be set to 2 for gridtype linear"
                x = np.meshgrid(np.linspace(-1, 1, img_shape_x))
                y = np.meshgrid(np.linspace(-1, 1, img_shape_y))
                grid_x, grid_y = np.meshgrid(y, x)
                grid = np.stack((grid_x, grid_y), axis = 0)
            elif params.gridtype == 'sinusoidal':
                assert params.N_grid_channels == 4, "N_grid_channels must be set to 4 for gridtype sinusoidal"
                x1 = np.meshgrid(np.sin(np.linspace(0, 2*np.pi, img_shape_x)))
                x2 = np.meshgrid(np.cos(np.linspace(0, 2*np.pi, img_shape_x)))
                y1 = np.meshgrid(np.sin(np.linspace(0, 2*np.pi, img_shape_y)))
                y2 = np.meshgrid(np.cos(np.linspace(0, 2*np.pi, img_shape_y)))
                grid_x1, grid_y1 = np.meshgrid(y1, x1)
                grid_x2, grid_y2 = np.meshgrid(y2, x2)
                grid = np.expand_dims(np.stack((grid_x1, grid_y1, grid_x2, grid_y2), axis = 0), axis = 0)
            img = np.concatenate((img, grid), axis = 1 )

    if params.roll:
        img = np.roll(img, y_roll, axis = -1)

    if train and (crop_size_x or crop_size_y):
        img = img[:,rnd_x:rnd_x+crop_size_x, rnd_y:rnd_y+crop_size_y]

    img = np.reshape(img, (n_channels, crop_size_x, crop_size_y))
    return torch.as_tensor(img)
