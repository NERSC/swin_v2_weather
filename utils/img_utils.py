import h5py
import math
import torchvision.transforms as T
import matplotlib
import matplotlib.pyplot as plt

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

