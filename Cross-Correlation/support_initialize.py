import numpy as np
import os
import torch
from torch import nn
from tqdm import tqdm
from support_models import generate_fake_sample
from datetime import datetime
from support_deal import preprocess
from torch.utils.tensorboard import SummaryWriter

log_dir = os.path.join(r'D:\Zang\Deaplearn_sec_Tensorborad', datetime.now().strftime("%Y%m%d-%H"))
writer = SummaryWriter(log_dir=log_dir)

def initialize_train(model, batch_size, num_epochs, lr, weight_decay):
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)
    pbar = tqdm(range(num_epochs))


    # used in generate(real) params to 0-1 zone
    def Params_norm(params, norm_params):

        params[:, 0] = params[:, 0] / norm_params.squeeze(1)
        params[:, 1] = -0.25 * (1 + torch.log10(params[:, 1]))
        params[:, 2] = (params[:, 2] - 0.1) / 0.216
    
        return params
    
    
    
    val_data, val_params_true = generate_fake_sample(batch_size, 0.15)
    val_data, val_params_true = torch.from_numpy(val_data).float().cuda(), torch.from_numpy(val_params_true).float().cuda()
    # Val_data is general data
    val_norm_data, Val_norm_value = preprocess(val_data)
    # val_norm_data is data norm to 0-1, Val_norm_params is the norm parameters, all data
    val_norm_params = Params_norm(val_params_true, Val_norm_value)
    val_attention_mask = torch.zeros(val_data.shape[:-1]).cuda()
    
    best = {'loss': 999, 'model': None}
    
    for i, epoch in enumerate(pbar):
        nosiegain = 0.05 + (0.10 * ((i+1)/ num_epochs ))
        data, params_true = generate_fake_sample(batch_size, nosiegain)
        attention_mask = torch.zeros(data.shape[:-1]).cuda()
        data, params_true = torch.from_numpy(data).float().cuda(), torch.from_numpy(params_true).float().cuda()
        norm_data, data_norm_value = preprocess(data)
        tarin_norm_params = Params_norm(params_true, data_norm_value)
        optimizer.zero_grad()
        outputs_params = model(norm_data, attention_mask)
        loss = criterion(outputs_params, tarin_norm_params)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if i % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_params_pred = model(val_norm_data, val_attention_mask)
                val_loss = criterion(val_params_pred, val_norm_params)
                if val_loss < best['loss']:
                    best['loss'] = val_loss
                    best['model'] = model.state_dict()
            model.train()
        writer.add_scalars("loss", {"loss@train": loss.cpu().item(), "loss@val": val_loss.cpu().item()}, epoch)
        pbar.set_description(f'epoch: {epoch}, loss: {loss.item():.4f} / {val_loss.item():.4f}')
    print(f'best loss: {best["loss"]:.4f}.')
    model.load_state_dict(best['model'])
    return model
