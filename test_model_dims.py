import torch
from networks.afno import AFNONet
from utils.YParams import YParams
from torchinfo import summary

params = YParams('./config/AFNO.yaml', 'afno')
model = AFNONet(params)
summary(model, input_size=(1,26,720,1440))
