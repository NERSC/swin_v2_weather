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

import torch
import time
import numpy as np
import h5py
import scipy.ndimage as ndi
from skimage.transform import downscale_local_mean
import matplotlib
import matplotlib.pyplot as plt

def interpolate_skimage(img, scale):
#    new_img = ndi.zoom(img, (1, 1, 1/scale[0], 1/scale[1]))
    new_img = downscale_local_mean(img, (1, 1, scale[0], scale[1]))
    # check images
#    plt.rcParams["figure.figsize"] = (20,20)
#    plt.figure()
#    time_idxes = [0, 1, 3, 4]
#    for i in time_idxes:
#        for ch in range(new_img.shape[1]):
#            plt.subplot(new_img.shape[1],1, ch+1)
#            plt.imshow(new_img[i,ch,:,:], cmap = 'RdBu')
#            plt.colorbar()
#        plt.savefig("pdfs/datastats_" + str(i) + ".jpg")
    return new_img

#years = [1979] 
years = [1979, 1989, 1999, 2004, 2010]


interp_factor_x = 2
interp_factor_y = 2
scale = (interp_factor_x, interp_factor_y)

img_shape_x = 720//interp_factor_x+1
img_shape_y = 1440//interp_factor_y
n_ch = 34

global_means = np.zeros((1,n_ch,1,1))
global_stds = np.zeros((1,n_ch,1,1))
time_means = np.zeros((1,n_ch,img_shape_x,img_shape_y))

base_path = "/pscratch/sd/j/jpathak/34var/"

time_steps = 500

for ii, year in enumerate(years):
    t0 = time.time()
    with h5py.File(base_path + '/train/'+ str(year) + '.h5', 'r') as f:
        rnd_idx = np.random.randint(0, 1460-time_steps)
        field = f['fields'][rnd_idx:rnd_idx+time_steps]
        if interp_factor_x != 1 or interp_factor_y != 1:
            # down/up sample image
            field = interpolate_skimage(field, scale)
        print(field.shape)
        global_means += np.mean(field, keepdims=True, axis = (0,2,3))
        global_stds += np.var(field, keepdims=True, axis = (0,2,3))
        time_means += np.mean(field, keepdims=True, axis = (0))
    print("time for year {} = {}".format(year, time.time() - t0))

global_means = global_means/len(years)
global_stds = np.sqrt(global_stds/len(years))
time_means = time_means/len(years)

np.save(base_path + '/new_stats/' + '/global_means_50km.npy', global_means)
np.save(base_path + '/new_stats/' + '/global_stds_50km.npy', global_stds)
np.save(base_path + '/new_stats/' + '/time_means_50km.npy', time_means)

# check stats
#gm = np.load("/pscratch/sd/j/jpathak/34var/new_stats/global_means.npy")
#gs = np.load("/pscratch/sd/j/jpathak/34var/new_stats/global_stds.npy")
#tm = np.load("/pscratch/sd/j/jpathak/34var/new_stats/time_means.npy")


print("Finished")
#print("means: ", global_means)
#print("stds: ", global_stds)







