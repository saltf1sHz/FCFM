import os
import torch
import pandas as pd
import numpy as np

def minmax_norm(y):

    y_min, y_max = y.min(axis=-1, keepdims=True).values, y.max(axis=-1,keepdims=True).values
    norm_params = (y_max - 1)
    y_prime = (y - 1) / norm_params

    return y_prime, norm_params

def preprocess(data):
    if len(data.shape) == 3:
        data[:, :, 1], norm_params = minmax_norm(data[:, :, 1])
    elif len(data.shape) == 2:
        data[:, 1], norm_params = minmax_norm(data[:, 1])
    # data = data + 1e-12
    return [data, norm_params]


def minmax_denorm(y_prime, norm_params):
    y_origin = (y_prime * norm_params) + 1
    return y_origin


def load_data(path):
    df = pd.read_csv(path, header=None)

    return torch.from_numpy(df.to_numpy())

def load_all_data(folder_path):
    data = []
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            file_path = os.path.join(folder_path, file)
            data.append(load_data(os.path.join(folder_path, file)))
    return data

def data_filter(data):

    # data = data.cpu().numpy()
    data = data[data[:,0] > 1e-7]
    filtered_data = data

    return torch.from_numpy(filtered_data).float().cuda()


def collate_fn(sample_list):
    sample_list = [data_filter(x) for x in sample_list]
    sample_list = [preprocess(x)[0] for x in sample_list] # BUG: Loss
    max_seq_len = max([x.shape[0] for x in sample_list])
    for i, sample in enumerate(sample_list):
        sample_list[i] = torch.cat([sample, torch.zeros(max_seq_len - sample.shape[0], 2).to('cuda')])
    batch_sample = torch.stack(sample_list) # (bsz, max_seq_len, 2)
    attention_mask = (batch_sample != 0).any(axis=2)*1
    # attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)# wzz
    attention_mask = (1.0 - attention_mask) * -10000.0
    return batch_sample, attention_mask