import numpy as np
import torch

def lat_np(j, num_lat):
    return 90 - j * 180/(num_lat-1)
# torch version for rmse comp

@torch.jit.script
def lat(j: torch.Tensor, num_lat: int) -> torch.Tensor:
    return 90. - j * 180./float(num_lat-1)

@torch.jit.script
def latitude_weighting_factor_torch(j: torch.Tensor, num_lat: int, s: torch.Tensor) -> torch.Tensor:
    return num_lat * torch.cos(3.1416/180. * lat(j, num_lat))/s

@torch.jit.script
def weighted_rmse_torch_channels(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    #takes in arrays of size [n, c, h, w]  and returns latitude-weighted rmse for each channel
    num_lat = pred.shape[2]
    lat_t = torch.arange(start=0, end=num_lat, device=pred.device)

    s = torch.sum(torch.cos(3.1416/180. * lat(lat_t, num_lat)))
    weight = torch.reshape(latitude_weighting_factor_torch(lat_t, num_lat, s), (1, 1, -1, 1))
    # Ensure the weights are non-negative
    # print(weight)
    # print(torch.where(weight < 0))
    # assert torch.all(weight >= 0), "Negative weights encountered,"
    
    result = torch.sqrt(torch.mean(weight * (pred - target)**2., dim=(-1,-2)))
    return result


def weighted_acc(pred,target, weighted  = True):
    # takes in shape [1, num_lat, num_long]
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

def latitude_weighting_factor_numpy(j: np.ndarray, num_lat: int, s: np.ndarray) -> np.ndarray:
    return num_lat * np.cos(3.1416/180. * lat_numpy(j, num_lat))/s

@torch.jit.script
def weighted_acc_masked_torch_channels(pred: torch.Tensor, target: torch.Tensor, maskarray: torch.Tensor) -> torch.Tensor:
    #takes in arrays of size [n, c, h, w]  and returns latitude-weighted acc
    num_lat = pred.shape[2]
    lat_t = torch.arange(start=0, end=num_lat, device=pred.device)
    s = torch.sum(torch.cos(3.1416/180. * lat(lat_t, num_lat)))
    weight = torch.reshape(latitude_weighting_factor_torch(lat_t, num_lat, s), (1, 1, -1, 1))
    result = torch.sum(maskarray * weight * pred * target, dim=(-1,-2)) / torch.sqrt(torch.sum(maskarray * weight * pred * pred, dim=(-1,-2)) * torch.sum(maskarray * weight * target *  target, dim=(-1,-2)))
    return result


def latitude_weighting_factor_numpy(j: np.ndarray, num_lat: int, s: np.ndarray) -> np.ndarray:
    # Using the absolute value to ensure non-negative weights
    return num_lat * np.abs(np.cos(np.pi/180. * lat_np(j, num_lat))) / s
def weighted_spread(pred: np.ndarray) -> np.ndarray:
    """
    Takes in arrays of size [n_ics, n_timesteps, n_channels, h, w] and returns 
    latitude-weighted spread for each channel per timestep.
    
    Args:
    pred (np.ndarray): A 5D numpy array of predictions with shape [n_ics, n_timesteps, n_channels, h, w].

    Returns:
    np.ndarray: A 2D numpy array of weighted average variances for each channel per timestep.
    """
    pred = pred.astype(np.float64)
    n_ics, n_timesteps, n_channels, num_lat, _ = pred.shape
    #takes in arrays of size [n, c, h, w]  and returns latitude-weighted rmse for each channel
    lat_t = np.arange(0,num_lat)
    s = np.sum(np.cos(3.1416/180. * lat_np(lat_t, num_lat)))
    weight = np.reshape(latitude_weighting_factor_numpy(lat_t, num_lat, s), (1, 1, -1, 1))
    
    # prediction variance at each grid point for each channel and timestep
    variance_per_grid_point = np.var(pred, axis=0)  # Dimension [n_timesteps, n_channels, h, w]
    
    # latitude weighting to the variance
    weighted_variance = weight * variance_per_grid_point
    
    # Average the weighted variances over all grid points, for each timestep and channel
    weighted_average_variance = np.mean(weighted_variance, axis=(-1, -2))  # Averaging over latitude and longitude
    
    # Check for negative variances
    if np.any(weighted_average_variance < 0):
        raise ValueError("Negative weighted variances encountered.")

    return weighted_average_variance

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
