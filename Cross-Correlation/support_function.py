import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

def Origin_CorrelationFunction(x, G0, Td, s):

    t1 = (1 / (1 + (x / Td)))
    t2 = (1 / torch.pow((1+((torch.pow(s, 2)) * (x / Td))), (0.5)))
    t3 = G0
    y = (t1 * t2 * t3 ) + 1

    return y

def Norm_CorrelationFunction(x, G0, Td, s):

    t1 = (1 / (1 + (x / (Td +1e-8))))
    t2 = (1 / torch.pow((1+((torch.pow(s, 2)) * (x / (Td +1e-8)))), (0.5)))
    t3 = G0
    y = (t1 * t2 * t3)

    return y

        
def Params_denorm(params):
    # params = torch.tensor(params)
    params[:, 1] = torch.pow(10, (-1-4*params[:, 1]))
    params[:, 2] = (params[:, 2] * 0.216) + 0.1
    
    return params


