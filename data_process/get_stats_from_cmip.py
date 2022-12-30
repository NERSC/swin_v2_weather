import torch
import time
import numpy as np
import h5py
import scipy.ndimage as ndi
from skimage.transform import downscale_local_mean
import matplotlib
import matplotlib.pyplot as plt

def interpolate_skimage(img, scale):
    new_img = downscale_local_mean(img, (1, 1, scale[0], scale[1]))
    return new_img

def concat(pl, sl):
    ''' pl is (nt, n_var, n_l, ix, iy)
        sl is (nt, n_var, ix, iy)
    '''
    pl_shape = pl.shape
    sl_shape = sl.shape
    pllist = []
    for i in range(pl_shape[1]): # for each variable 
        pllist.append(pl[:, i, ...])
    pl = np.concatenate(pllist, axis=1)
    tensor = np.concatenate([pl, sl], axis=1)
    return tensor

#years = [1979] 
years = [1979, 1989, 1999, 2004, 2010]

orig_size_x = 361
orig_size_y = 720

img_shape_x = orig_size_x
img_shape_y = orig_size_y #//interp_factor_y
n_ch = 39

global_means = np.zeros((1,n_ch,1,1))
global_stds = np.zeros((1,n_ch,1,1))
time_means = np.zeros((1,n_ch,img_shape_x,img_shape_y))

base_path = "/pscratch/sd/p/pharring/cmip_data/ECMWF-IFS-HR/r1i1p1f1"

time_steps = 500

for ii, year in enumerate(years):
    t0 = time.time()
    with h5py.File(base_path + '/train/'+ str(year) + '.h5', 'r') as f:
        rnd_idx = np.random.randint(0, 1460-time_steps)
        pl = f['pl'][rnd_idx:rnd_idx+time_steps]
        sl = f['sl'][rnd_idx:rnd_idx+time_steps]
        field = concat(pl, sl)
        global_means += np.mean(field, keepdims=True, axis = (0,2,3))
        global_stds += np.var(field, keepdims=True, axis = (0,2,3))
        time_means += np.mean(field, keepdims=True, axis = (0))
    print("time for year {} = {}".format(year, time.time() - t0))

global_means = global_means/len(years)
global_stds = np.sqrt(global_stds/len(years))
time_means = time_means/len(years)

np.save(base_path + '/stats/' + '/global_means.npy', global_means)
np.save(base_path + '/stats/' + '/global_stds.npy', global_stds)
np.save(base_path + '/stats/' + '/time_means.npy', time_means)

print("finished")







