import numpy as np
import torch

def lat_np(j, num_lat):
    return 90 - j * 180/(num_lat-1)

def weighted_acc(pred,target, weighted  = True):
    #takes in shape [1, num_lat, num_long]
    if len(pred.shape) ==2:
        pred = np.expand_dims(pred, 0)
    if len(target.shape) ==2:
        target = np.expand_dims(target, 0)
    
    num_lat = np.shape(pred)[1]
    num_long = np.shape(target)[2]
    s = np.sum(np.cos(np.pi/180* lat_np(np.arange(0, num_lat), num_lat)))
    weight = np.expand_dims(latitude_weighting_factor(np.arange(0, num_lat), num_lat, s), -1) if weighted else 1
    r = (weight*pred*target).sum() /np.sqrt((weight*pred*pred).sum() * (weight*target*target).sum())
    return r

def weighted_rmse(pred, target):
    if len(pred.shape) ==2:
        pred = np.expand_dims(pred, 0)
    if len(target.shape) ==2:
        target = np.expand_dims(target, 0)
    #takes in arrays of size [1, h, w]  and returns latitude-weighted rmse
    num_lat = np.shape(pred)[1]
    num_long = np.shape(target)[2]
    s = np.sum(np.cos(np.pi/180* lat_np(np.arange(0, num_lat), num_lat)))
    weight = np.expand_dims(latitude_weighting_factor(np.arange(0, num_lat), num_lat, s), -1)
    return np.sqrt(1/num_lat * 1/num_long  * np.sum(np.dot(weight.T, (pred[0] - target[0])**2)))

def latitude_weighting_factor(j, num_lat, s):
    return num_lat*np.cos(np.pi/180. * lat_np(j, num_lat))/s

def top_quantiles_error(pred, target):
    if len(pred.shape) ==2:
        pred = np.expand_dims(pred, 0)
    if len(target.shape) ==2:
        target = np.expand_dims(target, 0)
    qs = 100
    qlim = 5
    qcut = 0.1
    qtile = 1. - np.logspace(-qlim, -qcut, num=qs)
    P_tar = np.quantile(target, q=qtile, axis=(1,2))
    P_pred = np.quantile(pred, q=qtile, axis=(1,2))
    return np.mean(P_pred - P_tar, axis=0)


# torch version for rmse comp
@torch.jit.script
def lat(j: torch.Tensor, num_lat: int) -> torch.Tensor:
    return 90. - j * 180./float(num_lat-1)

@torch.jit.script
def latitude_weighting_factor_torch(j: torch.Tensor, num_lat: int, s: torch.Tensor) -> torch.Tensor:
    return num_lat * torch.cos(3.1416/180. * lat(j, num_lat))/s

@torch.jit.script
def weighted_rmse_torch_channels(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    #takes in arrays of size [n, c, h, w]  and returns latitude-weighted rmse for each chann
    num_lat = pred.shape[2]
    #num_long = target.shape[2]
    lat_t = torch.arange(start=0, end=num_lat, device=pred.device)

    s = torch.sum(torch.cos(3.1416/180. * lat(lat_t, num_lat)))
    weight = torch.reshape(latitude_weighting_factor_torch(lat_t, num_lat, s), (1, 1, -1, 1))
    result = torch.sqrt(torch.mean(weight * (pred - target)**2., dim=(-1,-2)))
    return result

@torch.jit.script
def weighted_rmse_torch_channels_fld(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    #takes in arrays of size [n, c, h, w]  and returns latitude-weighted rmse for each chann
    num_lat = pred.shape[2]
    #num_long = target.shape[2]
    lat_t = torch.arange(start=0, end=num_lat, device=pred.device)

    s = torch.sum(torch.cos(3.1416/180. * lat(lat_t, num_lat)))
    weight = torch.reshape(latitude_weighting_factor_torch(lat_t, num_lat, s), (1, 1, -1, 1))
    result = torch.sqrt(weight * (pred - target)**2.)
    return result

@torch.jit.script
def weighted_rmse_torch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    result = weighted_rmse_torch_channels(pred, target)
    return torch.mean(result, dim=0)


@torch.jit.script
def weighted_acc_torch_channels(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    #takes in arrays of size [n, c, h, w]  and returns latitude-weighted acc
    num_lat = pred.shape[2]
    #num_long = target.shape[2]
    lat_t = torch.arange(start=0, end=num_lat, device=pred.device)
    s = torch.sum(torch.cos(3.1416/180. * lat(lat_t, num_lat)))
    weight = torch.reshape(latitude_weighting_factor_torch(lat_t, num_lat, s), (1, 1, -1, 1))
    result = torch.sum(weight * pred * target, dim=(-1,-2)) / torch.sqrt(torch.sum(weight * pred * pred, dim=(-1,-2)) * torch.sum(weight * target *
    target, dim=(-1,-2)))
    return result

@torch.jit.script
def weighted_acc_torch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    result = weighted_acc_torch_channels(pred, target)
    return torch.mean(result, dim=0)

@torch.jit.script
def unweighted_acc_torch_channels(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    result = torch.sum(pred * target, dim=(-1,-2)) / torch.sqrt(torch.sum(pred * pred, dim=(-1,-2)) * torch.sum(target *
    target, dim=(-1,-2)))
    return result

@torch.jit.script
def unweighted_acc_torch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    result = unweighted_acc_torch_channels(pred, target)
    return torch.mean(result, dim=0)

@torch.jit.script
def top_quantiles_error_torch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    qs = 100
    qlim = 3
    qcut = 0.1
    n, c, h, w = pred.size()
    qtile = 1. - torch.logspace(-qlim, -qcut, steps=qs, device=pred.device)
    P_tar = torch.quantile(target.view(n,c,h*w), q=qtile, dim=-1)
    P_pred = torch.quantile(pred.view(n,c,h*w), q=qtile, dim=-1)
    return torch.mean(P_pred - P_tar, dim=0)
