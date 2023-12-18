import torch
import numpy as np

def preprocess(params, data):
    if params.add_zenith:
        # data has inp, tar, izen, tzen
        inp, tar, izen, tzen = data
        inp = torch.cat([inp, izen], dim=1)  # Concatenate input with zenith angle
        return inp, tar
    else:
        return data
