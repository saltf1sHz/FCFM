import os
import torch
from torch import nn
import pandas as pd
import torch.optim as optim
from tqdm import tqdm
import random
import numpy as np
from scheduler import OptimizerParamScheduler
import argparse
from datetime import datetime
from models import CrazyThursdayKFC
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from support_function import Norm_CorrelationFunction, Params_denorm
from support_models import generate_fake_sample
from support_Loss import Cl_Loss, MSE_Loss
from support_deal import load_all_data, preprocess, data_filter, collate_fn
from support_initialize import initialize_train
from torch.utils.tensorboard import SummaryWriter

log_dir = os.path.join(r'D:\Zang\Deaplearn_Tensorborad', datetime.now().strftime("%Y%m%d-%H"))
writer = SummaryWriter(log_dir=log_dir)

class BertConfig(object):
    def __init__(self,
                hidden_size=32,
                max_position_embeddings=1024,
                num_hidden_layers=4, 
                num_attention_heads=4,
                intermediate_size=8*4,
                hidden_act="gelu", 
                hidden_dropout_prob=0.0,
                attention_probs_dropout_prob=0.0,
                initializer_range=0.02, 
                input_size=2,
                output_size=6
                ):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.output_size= output_size
        self.input_size = input_size
        self.max_position_embeddings = max_position_embeddings

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class AtomDataset(Dataset):
    def __init__(self, data_array) -> None:
        self.data = data_array
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        return self.data[index]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('-fpath', '--folder_path', type=str, default=r"D:\Zang\deeplearn_temp\Part1_data\Data")
    parser.add_argument('-fpath', '--folder_path', type=str, default=r"C:\Users\lonel\OneDrive\软件\DeepLearn\valdata\test_data")
    parser.add_argument('-bsz', '--batch_size', type=int, default=64)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4)
    parser.add_argument('-ep', '--max_epoch', type=int, default=1600)
    parser.add_argument('-wd','--weight_decay', type=float, default=1e-5)
    parser.add_argument('-sd','--seed', type=int, default=2024)

    parser.add_argument('-ibsz', '--init_batch_size', type=int, default=64)
    parser.add_argument('-ilr', '--init_learning_rate', type=float, default=1e-3)
    parser.add_argument('-iep', '--init_max_epoch', type=int, default=1600)
    parser.add_argument('-iwd','--init_weight_decay', type=float, default=5e-5)
    args = parser.parse_args()
    setup_seed(args.seed)

    VAL_RATIO = 0.25
    VAL_STEP = 50

    bert_config = BertConfig()
    model = CrazyThursdayKFC(bert_config)
    model.cuda()
    model = initialize_train(model, args.init_batch_size, args.init_max_epoch, args.init_learning_rate, args.init_weight_decay)
    model_initial_path =r"D:\Zang\deeplearn_temp\model_in.pkl"
    torch.save(model.state_dict(), model_initial_path)

    val_data, val_params = generate_fake_sample(args.batch_size, 0.15)
    val_data, val_params = torch.from_numpy(val_data).float().cuda(), torch.from_numpy(val_params).float().cuda()
    val_norm_inputs, val_norm_params = preprocess(val_data)

    opt = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = OptimizerParamScheduler(
        opt, 
        max_lr=args.learning_rate, 
        min_lr=args.learning_rate/100,
        lr_warmup_steps=int(args.max_epoch*0.01), 
        lr_decay_steps=int(args.max_epoch*0.75), 
        lr_decay_style='cosine',
        start_wd=args.weight_decay, 
        end_wd=args.weight_decay, 
        wd_incr_steps=1, 
        wd_incr_style='constant',
        use_checkpoint_opt_param_scheduler=True,
        override_opt_param_scheduler=False)

    best = dict(val_loss=9999999999999999)
    pbar = tqdm(range(args.max_epoch))
    for ep in pbar:
        nosiegain = 0.05 + (0.10 * ( (ep+1)/ args.max_epoch ))
        train_data, train_params = generate_fake_sample(args.batch_size, nosiegain)
        train_data, train_params = torch.from_numpy(train_data).float().cuda(), torch.from_numpy(train_params).float().cuda()
        norm_train_inputs, norm_params = preprocess(train_data)       
        train_attention_mask = torch.zeros(norm_train_inputs.shape[:-1]).cuda() # (bsz, seq_len)
        outputs_params = model(norm_train_inputs, train_attention_mask)
        denorm_outputs_params = Params_denorm(outputs_params)
        train_loss = Cl_Loss(norm_train_inputs[:,:,0], norm_train_inputs[:,:,1], denorm_outputs_params)
        if torch.isnan(train_loss):
            print(ep, train_loss.item())
            # print(real_params)
            import sys; sys.exit()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        train_loss.backward()
        
        opt.step()
        scheduler.step(1)

        if ep % VAL_STEP == 0:
            model.eval()
            with torch.no_grad():
                val_attention_mask = torch.zeros(val_norm_inputs.shape[:-1]).cuda()
                val_output_params = model(val_norm_inputs, val_attention_mask)
                print("Model output:", val_output_params)
                val_denorm_params = Params_denorm(val_output_params)
                print("Validation data:", val_norm_inputs)
                print("Normalized parameters:", val_denorm_params)
                val_loss = MSE_Loss(val_norm_inputs[:,:,0], val_norm_inputs[:,:,1], val_denorm_params)
            model.train()

        writer.add_scalars("loss", {"loss@train": train_loss.cpu().item(), "loss@val": val_loss.cpu().item()}, ep)
        pbar.set_description(f"loss@train:{train_loss.cpu().item():4f}, loss@val:{val_loss.cpu().item():4f}")
        
        if val_loss.cpu().item() < best['val_loss']:
            best = dict(train_loss=train_loss.cpu().item(), val_loss=val_loss.cpu().item(), epoch=ep)
            best['model_weight'] = model.state_dict()
            model_path =r"D:\Zang\deeplearn_temp\model_en.pkl"
            torch.save(best['model_weight'], model_path)

    for k, v in best.items():
        print(f"{k:10s}: {v}")

    model_path =r"D:\Zang\deeplearn_temp\model_en.pkl"
    torch.save(best['model_weight'], model_path)