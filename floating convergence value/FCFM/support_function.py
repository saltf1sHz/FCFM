import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

def Origin_CorrelationFunction(x, c, G0, Td, T, Tt, s, f):

    t1 = 1 + (T * torch.exp(-x / Tt) / (1- T))
    t2 = 1 / (1 + torch.pow((x / Td), f))
    t3 = 1 / torch.pow((1 + (torch.pow(s, 2)) * torch.pow(x / Td, f)), 0.5)
    y = G0 * t1 * t2 * t3 + c

    return y

def Norm_CorrelationFunction(x, c, G0, Td, T, Tt, s, f):

    # After norm the data is mapping0-1, so the +_1 was delteted
    t1 = 1 + (T * torch.exp(-x / Tt) / (1- T))
    t2 = 1 / (1 + torch.pow((x / Td), f))
    t3 = 1 / torch.pow((1 + (torch.pow(s, 2)) * torch.pow(x / Td, f)), 0.5)
    y = G0 * t1 * t2 * t3 + c

    return y

def Params_denorm(params):
    # params = torch.tensor(params)
    params[:, 2] = torch.pow(10, (-1-4*params[:, 2]))
    params[:, 4] =torch.pow(10, (-5-4*params[:, 4]))
    params[:, 5] = (params[:, 4] * 0.216) + 0.1
    
    return params
