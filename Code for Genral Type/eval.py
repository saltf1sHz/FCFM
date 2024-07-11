import os
import torch
from torch import nn
import pandas as pd
from tqdm import tqdm
import numpy as np
from models import CrazyThursdayKFC
from train import BertConfig
from support_function import  Norm_CorrelationFunction, Origin_CorrelationFunction
from matplotlib import pyplot as plt
from support_Loss import Cl_Loss, MSE_Loss
from support_deal import load_data, preprocess, data_filter, minmax_norm, minmax_denorm



if __name__ == "__main__":

    # model_path = r"D:\Zang\deeplearn_temp\model_Z_1125_cl.pkl"
    model_path = r"C:\Users\lonel\OneDrive\软件\DeepLearn\general_dl\model_en.pkl"
    # data_path = r"C:\Users\lonel\OneDrive\软件\DeepLearn\ACF验证数据\real_data\test_data\34.csv"
    data_path = r"C:\Users\lonel\OneDrive\软件\DeepLearn\general_data\22.csv"

    bert_config = BertConfig()
    model = CrazyThursdayKFC(bert_config)
    # model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=True)
    model.load_state_dict(torch.load(model_path, map_location='cuda'), strict=True)
    model = model.cuda()   
    model.eval()

    data = load_data(data_path)
    origindata = data.clone()
    deal_data, norm_params = preprocess((data))
    data = data.cuda()
    
    def Params_denorm(params, norm_params):
    # params = torch.tensor(params)
        params[0] = params[0] * norm_params
        params[1] = torch.pow(10, (-1-4*params[1]))
        params[3] =torch.pow(10, (-5-4*params[3]))
        params[4] = (params[4] * 0.216) + 0.1
        
        return params
    
    with torch.no_grad():
        
        deal_data = torch.tensor(deal_data).float().cuda()
        input_data = deal_data.unsqueeze(0).cuda()
        attention_mask = torch.zeros(input_data.shape[:-1]).cuda()
        # The six parameters predicted by the model.
        outputs_params = model(input_data, attention_mask).squeeze(0).cuda()
        outputs_params = outputs_params.cpu()
        real_params = Params_denorm(outputs_params, norm_params)
        # The y-sequence generated by the function with the six parameters above.
        y_prime = Origin_CorrelationFunction(input_data[:, :, 0], *real_params)
        # y_pre_pred, y_min, y_max = np.y_pre_pred, np.y_min, np.y_max

    N, Td, T, Tt, s, f = real_params.cpu().numpy()
    print(f'N: {N}, Td: {Td}, T: {T}, Tt: {Tt}, s: {s}, f: {f}')


    data = data.cpu().numpy()
        
    fig, ax = plt.subplots()

    input_data, y_prime = input_data.cpu().numpy(), y_prime.cpu().numpy()
    fitx = input_data[0, :, 0].squeeze()
    fity = y_prime.squeeze()

    ax.clear()
    ax.scatter(origindata[:, 0], origindata[:, 1], label="Original Data", s=8)
    ax.plot(fitx,  fity, label="Model Prediction", color='red')
    ax.set_xlabel("Input")
    ax.set_ylabel("Output")
    ax.set_xscale('log')
    ax.legend()
    # plt.title("MSE_Loss" + str(mse_loss.item()) + "  " + "ClLoss" + str(Cl_loss))
    plt.show()

    plt.close()