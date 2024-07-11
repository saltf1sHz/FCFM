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
import csv


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

class Fit_general_type(nn.Module):
    def __init__(self, ):
        super(Fit_general_type, self).__init__()
        """ self.G0 = nn.Parameter(torch.tensor([inG0], requires_grad=True)) """
        self.G0 = nn.Parameter(torch.rand(1, requires_grad=True))
        self.Td = nn.Parameter(torch.rand(1, requires_grad=True))
        # self.Td = nn.Parameter(torch.rand(1, requires_grad=True))
        # self.Td = nn.Parameter(torch.tensor([intd], requires_grad=False))
        self.T = nn.Parameter(torch.rand(1, requires_grad=True))
        # self.T = nn.Parameter(torch.tensor([inT], requires_grad=False))
        # self.f = nn.Parameter(torch.rand(1, requires_grad=True))
        self.Tt = nn.Parameter(torch.rand(1, requires_grad=True))
        # self.Tt = nn.Parameter(torch.tensor([inTt], requires_grad=False))
        self.s = nn.Parameter(torch.tensor([0.0], requires_grad=True))
        self.f = nn.Parameter(torch.rand(1, requires_grad=True))
        """ self.f = nn.Parameter(torch.tensor([0.0], requires_grad=True)) """

    def regular(self):
        """ G0 = torch.sigmoid(self.G0) """
        G0 = torch.abs(self.G0)
        Td = torch.sigmoid(self.Td)
        T = 2 * torch.abs(torch.sigmoid(self.T)-0.5)
        """ T = torch.abs(self.T-self.T)+ 0 """
        Tt = torch.sigmoid(self.Tt)

        s = 0.1 + 0.216 * torch.sigmoid(self.s)

        f = torch.sigmoid(self.f)

        return G0, Td, T, Tt, s, f
    

    def forward(self, x):
        G0, Td, T, Tt, s, f = self.regular()


        Tt = torch.pow(10, ((-4*Tt)-5))
        Td = torch.pow(10, ((-4*Td)-2))
        
        
        m1 = (1 + (T * (torch.exp(-x / Tt) / (1-T+1e-8))))
        m2 = (1 / (1 + torch.pow((x / Td), f)))
        m3 = (1 / torch.pow((1+((torch.pow(s, 2)) * torch.pow((x / Td), f))), (0.5)))
        m4 = G0
        
        y = m1 * m2 * m3 * m4
        
        return y
    
    def get_params(self):
        G0, Td, T, Tt, s, f = self.regular()
        G0, Td, T, Tt, s, f =G0.data.cpu().numpy()[0], Td.data.cpu().numpy()[0], T.data.cpu().numpy()[0], Tt.data.cpu().numpy()[0], s.data.cpu().numpy()[0], f.data.cpu().numpy()[0]
        return G0, Td, T,  Tt, s, f
    
    def set_params(self, G0, Td, T, Tt, s, f):
        if type(G0) == float:
            self.G0.data = torch.tensor([G0])
            self.Td.data = torch.tensor([Td])
            self.T.data = torch.tensor([T])
            self.Tt.data = torch.tensor([Tt])
            self.s.data = torch.tensor([s])
            self.f.data = torch.tensor([f])
            
        else:
            self.G0.data = G0
            self.Td.data = Td
            self.T.data = T
            self.Tt.data = Tt
            self.s.data = s
            self.f.data = f
            
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
    data_prime = (data - 1) / (data_max - 1)
    return data_prime, data_min, data_max

def minmax_denorm(data_prime, data_min, data_max):
    return (data_max - 1) * data_prime + 1



if __name__ =="__main__":
    record_path = r"C:\Users\lonel\OneDrive\s\MSE.csv"
    data_folder = r"C:\Users\lonel\OneDrive\soft\DeepLearn\general_data"
    # data_folder = r'C:\Users\lonel\OneDrive\软件\DeepLearn\low'
    os.chdir(data_folder)
    max_ep = 251
    sample_id = 13
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
    }
    
    mse_criterion = nn.MSELoss()
    
    setup_seed(seed)
    data = load_data(f"{sample_id}.csv")
    filtered_data = data
    
    # filtered_data, give_params = data_filter(data)

    x, y = filtered_data[:,0],filtered_data[:,1]
    """ y_pred = np.array(results['y_pred']).flatten() """
    y_prime, y_min, y_max = minmax_norm(y)
    y_pred = minmax_denorm(y_prime, y_min, y_max)
    filtered_data[:,0],filtered_data[:,1] = x, y_prime

    model = Fit_general_type()
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
    
    
    with open(record_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Training Loss', 'MSE Loss'])
    
    

    for ep in pbar:
        
        outputs = model(inputs)
        for opt in opt_list:
            opt.zero_grad()
        loss = loss_function(outputs, labels)
        mse_loss = mse_criterion(outputs, labels)
        
        if torch.isnan(loss):
            print(model.get_params())
            print(loss)
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for opt in opt_list:
            opt.step()
        
        with open(record_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([ep, loss.cpu().item(), mse_loss.cpu().item()]) 
        
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

    norm_G0, Td, T, Tt, s, f = best['params']
    
    Tt = 10 ** ((-4*Tt)-5)
    Td = 10 ** ((-4*Td)-2)

    G0 = (np.array(y_max)-1) * norm_G0 + 1

    print(f"Params:G0:{G0:.6f}, Td:{Td}, T:{T:.6f}, Tt:{Tt}, s:{s:.6f}, f:{f:.6f}, sample_id:{sample_id}")


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

    outputs_path = r'C:\Users\lonel\OneDrive\fit result\1.txt'
    
    outputs_cpu = outputs.cpu().detach().numpy() * norm_G0 + 1
    x_cpu = x.cpu().numpy()
    labels_cpu = labels.cpu().numpy() * norm_G0 + 1
    combined_data = np.column_stack((x_cpu, labels_cpu, outputs_cpu))
    np.savetxt(outputs_path, combined_data, fmt="%.9f")
    labels = labels.cpu().numpy()
    r2 = r2_score(labels, outputs_cpu)
    """ num_samples, num_features = inputs.shape
    adjusted_r2 = 1 - (1 - r2) * (num_samples - 1) / (num_samples - num_features - 1) """
    print(f"R² Score: {r2:.4f}")
    
    input("Press Enter to exit...")
    plt.close()
