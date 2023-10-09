import torch
import torch.nn as nn
from functools import partial
# networks
from networks.afno import AFNONet
from networks.swinv2_global import swinv2net
from networks.vit import ViT

class SingleStepWrapper(nn.Module):
    """Wrapper for training a single step into the future"""
    def __init__(self, params, model_handle):
        super(SingleStepWrapper, self).__init__()
        self.model = model_handle(params)

    def forward(self, inp):
        y = self.model(inp)
        return y


class MultiStepWrapper(nn.Module):
    """Wrapper for training multiple steps into the future"""
    def __init__(self, params, model_handle):
        super(MultiStepWrapper, self).__init__()
        self.model = model_handle(params)
        self.n_future = params.n_future

    def forward(self, inp):
        result = []
        inpt = inp
        for step in range(self.n_future + 1):
            pred = self.model(inpt)
            result.append(pred)
            if step == self.n_future:
                break
            inpt = pred
        result = torch.cat(result, dim=1)
        return result

def get_model(params):
    if params.nettype == 'afno':
        model = partial(AFNONet)
    elif params.nettype == 'swin':
        model = partial(swinv2net)
    elif params.nettype == 'vit':
        model = partial(ViT)
    else:
        raise Exception(f"model type {params.nettype} not implemented")

    # wrap into Multi-Step if requested
    if params.n_future > 0:
        model = MultiStepWrapper(params, model)
    else:
        model = SingleStepWrapper(params, model)

    return model
