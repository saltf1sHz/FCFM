import torch
import numpy as np
import csv

def generate_fake_sample(batch_size, nosie_gain):
    def generate_log_uniform_range(log_range, num):
        return 10 ** np.random.uniform(*log_range, num)
    G0_range = (-3, 0)
    Td_range = (-5, -2)
    s_range = (0.1, 0.316)
    
    G0 = generate_log_uniform_range(G0_range, (batch_size,1)) # (batch_size, 1)
    Td = generate_log_uniform_range(Td_range, (batch_size,1))
    s = np.random.uniform(*s_range, (batch_size,1))
    
    params = np.hstack([G0, Td, s])

    num_points = np.random.randint(180, 300)
    x = np.logspace(np.log10(1e-5), np.log10(10), num_points)
    x = np.tile(x, (batch_size, 1)) # (batch_size, 900) 
    
    m1 = (1 / (1 + (x / Td)))
    m2 = (1 / ((1+((s ** 2) * (x / Td))) ** (0.5)))
    m3 = G0
    
    y = (m1 * m2 * m3) + 1

    def error_factor(y, nosie_gain):
        random_error = nosie_gain * np.random.normal(-1, 1, y.shape) * (y-1)
        y_nosie = y + random_error
        return y_nosie

    y_nosie = error_factor(y, nosie_gain)
    
    # data (64, 900, 2)
    data = np.stack([x, y_nosie], axis=-1)
    return data, params

if __name__ == "__main__":

    num_samples = 128

    params_filename = "parameters.csv"
    with open(params_filename, 'w', newline='') as params_file:
        params_writer = csv.writer(params_file)
        
        for i in range(num_samples):
            data, params = generate_fake_sample(1, 0.1)
            data = np.squeeze(data)
            params = np.squeeze(params)
            # print(data)
            params_writer.writerow([params])
            filename = f"{i + 1}.csv"
            with open(filename, 'w', newline='') as csvfile:
                x, y_nosie = data[:, 0], data[:, 1]
                csvwriter = csv.writer(csvfile)
                csvwriter.writerows(zip(x, y_nosie))
