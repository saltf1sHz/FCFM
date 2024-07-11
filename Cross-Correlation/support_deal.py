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
