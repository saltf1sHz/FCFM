import torch
from torch import nn
import numpy as np
from scipy.ndimage import gaussian_filter1d
from torch.utils.data import Dataset, DataLoader
from support_function import Origin_CorrelationFunction, Norm_CorrelationFunction

class CorrelationLoss(nn.Module):
    def __init__(self):
        super(CorrelationLoss, self).__init__()

    def forward(self, output, target):
        squared_difference = torch.pow((100*(output - target)), 2) * (torch.abs(1-target))
        weighted_difference = squared_difference 
        CorrelationLoss = torch.mean(weighted_difference)

        return CorrelationLoss

def Cl_Loss(x, y, params):
    criterion = CorrelationLoss()
    params = [params[:, i].unsqueeze(1) for i in range(params.squeeze().shape[1])]
    y_pred = Norm_CorrelationFunction(x, *params)
    return criterion(y_pred, y)

def MSE_Loss(x, y, params):
    # criterion = CustomMSELoss()
    criterion = nn.MSELoss()
    params = [params[:, i].unsqueeze(1) for i in range(params.squeeze().shape[1])]
    y_pred = Norm_CorrelationFunction(x, *params)
    return criterion(y_pred, y)

