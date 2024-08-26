import torch
import torch.nn as nn
import pandas as pd
import torch.optim as optim
from tqdm import tqdm
import random
import json
import numpy as np
from matplotlib import pyplot as plt
import time
import os
from torch.autograd import Variable
from sklearn.metrics import r2_score


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class CorrelationLoss(nn.Module):
    def __init__(self):
        super(CorrelationLoss, self).__init__()

    def forward(self, output, target):
        squared_difference = torch.pow((100*(output - target)), 2) * (torch.abs(1-target))
        weighted_difference = squared_difference 
        CorrelationLoss = torch.mean(weighted_difference)

        return CorrelationLoss
    
""" class WeightedMSELoss(nn.Module):
    def __init__(self):
        super(WeightedMSELoss, self).__init__()

    def forward(self, output, target):
        
        weights = 1.0 / variances

        WMSELoss = torch.mean(weights * (output - target) ** 2)

        return WMSELoss """


class Fit_general_type(nn.Module):
    def __init__(self, give_params):
        super(Fit_general_type, self).__init__()
        inG0, intd, inTt, inT, s, f = give_params
        """ self.G0 = nn.Parameter(torch.tensor([inG0], requires_grad=True)) """
        self.G0 = nn.Parameter(torch.rand(1, requires_grad=False))
        self.Td = nn.Parameter(torch.tensor([intd], requires_grad=True))
        # self.Td = nn.Parameter(torch.rand(1, requires_grad=True))
        # self.Td = nn.Parameter(torch.tensor([intd], requires_grad=False))
        self.T = nn.Parameter(torch.rand(1, requires_grad=True))
        # self.T = nn.Parameter(torch.tensor([inT], requires_grad=False))
        # self.f = nn.Parameter(torch.rand(1, requires_grad=True))
        self.Tt = nn.Parameter(torch.rand(1, requires_grad=True))
        # self.Tt = nn.Parameter(torch.tensor([inTt], requires_grad=False))
        self.s = nn.Parameter(torch.tensor([0.0], requires_grad=True))
        self.f = nn.Parameter(torch.rand(1, requires_grad=True))
        self.c = nn.Parameter(torch.rand(1, requires_grad=True))
        """ self.f = nn.Parameter(torch.tensor([0.0], requires_grad=True)) """

    def regular(self):
        """ G0 = torch.sigmoid(self.G0) """
        G0 = torch.abs(self.G0)
        Td = torch.abs(self.Td)
        T = torch.abs(torch.sigmoid(self.T)-0.5)
        """ T = torch.abs(self.T-self.T)+ 0 """
        Tt = torch.abs(self.Tt)
        """ Tt = torch.abs(self.T-self.T)+ 0 """
        s = 0.1 + 0.216 * torch.sigmoid(self.s)
        """ s = 0.6 + (1*(0.5-torch.sigmoid(self.s))) """
        """ s = 1-torch.abs(torch.sigmoid(self.s)-0.5) """
        """ s = (self.s-self.s)+0.55 """
        f = torch.sigmoid(self.f)
        """ f = torch.abs(self.f)/(torch.abs(self.f)+1) """
        """ f = torch.abs(torch.sigmoid(self.f)-0.5) """
        """ f = (self.f-self.f)+0.65231 """
        c = self.c 
        
        return G0, Td, T, Tt, s, f, c
    

    def forward(self, x):
        G0, Td, T, Tt, s, f, c = self.regular()


        Tt = Tt * 1e-5
        Td = Td * 1e-3
        
        
        m1 = (1 + (T * (torch.exp(-x / Tt) / (1-T+1e-8))))
        m2 = (1 / (1 + torch.pow((x / Td), f)))
        m3 = (1 / torch.pow((1+((torch.pow(s, 2)) * torch.pow((x / Td), f))), (0.5)))
        m4 = G0
        
        y = (m1 * m2 * m3 * m4) + c
        
        return y
    
    def get_params(self):
        G0, Td, T, Tt, s, f, c = self.regular()
        G0, Td, T, Tt, s, f, c =G0.data.cpu().numpy()[0], Td.data.cpu().numpy()[0], T.data.cpu().numpy()[0], Tt.data.cpu().numpy()[0], s.data.cpu().numpy()[0], f.data.cpu().numpy()[0], c.data.cpu().numpy()[0]
        return G0, Td, T,  Tt, s, f, c
    
    def set_params(self, G0, Td, T, Tt, s, f, c):
        if type(G0) == float:
            self.G0.data = torch.tensor([G0])
            self.Td.data = torch.tensor([Td])
            self.T.data = torch.tensor([T])
            self.Tt.data = torch.tensor([Tt])
            self.s.data = torch.tensor([s])
            self.f.data = torch.tensor([f])
            self.c.data = torch.tensor([c])
            
        else:
            self.G0.data = G0
            self.Td.data = Td
            self.T.data = T
            self.Tt.data = Tt
            self.s.data = s
            self.f.data = f
            self.c.data = c
            
def eval(outputs, labels):
    criterion = nn.MSELoss()
    loss = criterion(outputs, labels)
    return loss.cpu().item()

def get_opt_list(model, configs):
    opt_list = [optim.Adam([params], lr=configs[name][0], weight_decay=configs[name][1]) for name, params in model.named_parameters()]
    return opt_list

def load_data(path):
    df = pd.read_csv(path, header=None)
    return torch.from_numpy(df.to_numpy())

def decorate_params(model):
    params= model.get_params()
    _str = ""
    for x in params:
        _str += f"{x:.10f}, "
    return _str.strip().strip(",")

def minmax_norm(data):
    data_min, data_max = data.min(), data.max()
    data_prime = (data - data_min) / (data_max - data_min)
    return data_prime, data_min, data_max

def minmax_denorm(data_prime, data_min, data_max):
    return (data_max - data_min) * data_prime + data_min

def data_filter(data):

    def data_delete(data):
        data = data[data[:,0] > 1e-7]
        # data = data[data[:,0] > value_time]
        # data = data[data[:,1] > 0.99]
        return data
        
    def remove_outliers(data, threshold_multiplier=2):
        y_diff = np.diff(data[:, 1])

        mean_diff = np.mean(y_diff)
        std_diff = np.std(y_diff)
        threshold = mean_diff + threshold_multiplier * std_diff
        result = []
        indices_to_remove = []
        for i in range(len(data)):
            if i == 0 or abs(y_diff[i - 1]) <= threshold:
                result.append(data[i])
            else:
                indices_to_remove.append(i - 1) 

        filtered_data = np.delete(data, indices_to_remove, axis=0)

        return np.array(filtered_data)
    
    def remove_mean_outliers(data):
        y = data[:,1]
        mean = np.mean(y)
        std = np.std(y)
        threshold = mean + 4 * std
        filtered_data = data[data[:,1] < threshold]
        return filtered_data

    data = data_delete(data)
    filtered_data = remove_outliers(data)
    filtered_data = remove_mean_outliers(filtered_data)

    # filtered_data = np.array(data)

    inG0 = (np.mean(filtered_data[:10, 1]) - 1)

    inNP = (np.mean(filtered_data[:30, 1])+ 1)/2

    y_diff = np.abs(filtered_data[:, 1] - inNP)

    closest_index = np.argmin(y_diff)

    intd = filtered_data[closest_index, 0]*1e4

    """ def error_factor(y_values):
    # a selfmade error to increase reality
        random_factor = random.uniform(-2,2)
        return ((y_values-1) * random_factor) + y_values
    filtered_data[:,1 ]= error_factor(filtered_data[:,1 ]) """

    filtered_data = torch.tensor(filtered_data)

    give_params = [inG0, intd, 1e-9, 0.5, 0.55, 0.5]
    # give_params = [inN, intd, inT, inTt, s, f]
    print(give_params)
    return filtered_data, give_params



if __name__ =="__main__":
    data_folder = r'C:\Users\lonel\OneDrive\软件\DeepLearn\Transformer拟合FCS\general_data'
    # data_folder = r'C:\Users\lonel\OneDrive\软件\DeepLearn\low'
    os.chdir(data_folder)
    max_ep = 200
    sample_id = 6
    learning_rate = 8e-3
    weight_decay = 5e-5
    seed = 2023
    configs = {
        "G0":[learning_rate*5,weight_decay*25],
        "Td":[learning_rate*5,weight_decay*25], # [1e-5,weight_decay]
        "T":[learning_rate*5,weight_decay*5],
        "s":[learning_rate*50, weight_decay*15],
        "Tt":[learning_rate*10,weight_decay*15],
        "f":[learning_rate*50,weight_decay*5],
        "c":[learning_rate*2,weight_decay*2],
    }

    """ configs = {
        "G0":[learning_rate*50,weight_decay*25],
        "Td":[learning_rate*150,weight_decay*25], # [1e-5,weight_decay]
        "T":[learning_rate*10,weight_decay*50],
        "s":[learning_rate*10, weight_decay*5],
        "Tt":[learning_rate*5,weight_decay*250],
        "f":[learning_rate*1000,weight_decay*50],
    } """

    setup_seed(seed)
    data = load_data(f"{sample_id}.csv")

    filtered_data, give_params = data_filter(data)

    x, y = filtered_data[:,0],filtered_data[:,1]
    """ y_pred = np.array(results['y_pred']).flatten() """
    y_prime, y_min, y_max = minmax_norm(y)
    y_pred = minmax_denorm(y_prime, y_min, y_max)
    filtered_data[:,0],filtered_data[:,1] = x, y_prime

    model = Fit_general_type(give_params)
    model.cuda()

    """ loss_function = nn.MSELoss() """
    loss_function = CorrelationLoss()
    opt_list = get_opt_list(model, configs)
    pbar = tqdm(range(max_ep))
    inputs, labels = filtered_data[:,0], filtered_data[:,1]
    inputs, labels = inputs.cuda(),labels.cuda()
    
    plot_interval = 10
    fig, ax = plt.subplots()

    params_list = []
    best = dict(loss=970330)
    start_time = time.time()

    for ep in pbar:
        
        outputs = model(inputs)
        for opt in opt_list:
            opt.zero_grad()
        loss = loss_function(outputs, labels)
        if torch.isnan(loss):
            print(model.get_params())
            print(loss)
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for opt in opt_list:
            opt.step()
        if ep % plot_interval == 0:
            ax.clear() 
            ax.scatter(inputs.cpu().numpy(), labels.cpu().numpy(), label="Data", s=8)
            ax.plot(inputs.cpu().detach().numpy(), outputs.cpu().detach().numpy(), label="Model Prediction", color='red')
            ax.set_xlabel("Input")
            ax.set_ylabel("Output")
            ax.set_xscale('log')
            ax.legend()
            ax.set_title(f"Epoch {ep}, Loss: {loss.cpu().item():.5f}")
            plt.pause(0.1) 

        if loss.cpu().item() < best['loss']:
            best = dict(loss=loss.cpu().item(), epoch=ep, params=model.get_params())
        pbar.set_description(f"loss:{loss.cpu().item():6f} {decorate_params(model)}")
        if (ep + 1) % 100 == 0:
            params_list.append([float(x) for x in model.get_params()])
    
    end_time = time.time()
    fitting_time = end_time - start_time
    print(fitting_time)

    norm_G0, Td, T, Tt, s, f, c = best['params']
    
    Tt = Tt * 1e-5
    Td = Td * 1e-3
    
    y_min = y_min
    
    c = c * (np.array(y_max)-np.array(y_min)) + np.array(y_min)
    
    N = (np.array(y_max)-np.array(y_min)) * norm_G0 + (c-np.array(y_min))

    print(f"Params:N:{N:.6f}, Td:{Td}, T:{T:.6f}, Tt:{Tt}, s:{s:.6f}, f:{f:.6f}, c:{c:.6f}, sample_id:{sample_id}")


    with open("temp.json", 'w') as f:
        json.dump(params_list, f, indent=4)
    
        ax.clear()
        ax.scatter(data[:, 0], data[:, 1], label="Data", s=8)
        ax.plot(inputs.cpu().detach().numpy(), (outputs.cpu().detach().numpy() * (np.array(y_max)-1)) + 1, label="Model Prediction", color='red')
        ax.set_xlabel("Input")
        ax.set_ylabel("Output")
        ax.set_xscale('log')
        ax.legend()
        plt.title(f"Best Result - Epoch {best['epoch']}, Loss: {best['loss']:.6f}, Sample ID: {sample_id}")

        plt.show(block=False)

    outputs_cpu = outputs.cpu().detach().numpy()
    labels = labels.cpu().numpy()
    r2 = r2_score(labels, outputs_cpu)
    """ num_samples, num_features = inputs.shape
    adjusted_r2 = 1 - (1 - r2) * (num_samples - 1) / (num_samples - num_features - 1) """
    print(f"R² Score: {r2:.4f}")
    
    input("Press Enter to exit...")
    plt.close()