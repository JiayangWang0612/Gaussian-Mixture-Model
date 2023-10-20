import torch
import torch.nn as nn
from Model import *
from torch.optim import Adam
import numpy as np

      

def save_model(folder, k, n_feature, n_hidden, num_hidden_layers, n_component,
               model, optimizer, best=False):
    checkpoint = {'n_feature': n_feature,
                  'n_hidden': n_hidden,
                  'num_hidden_layers': num_hidden_layers,
                  'n_component': n_component,
                  'state_dict': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'epoch': k}
    if best:
        k = "best"
    torch.save(checkpoint, folder + str(k) + '_' + 'checkpoint.pth')


def load_model(file_path, device):
    checkpoint = torch.load(file_path)
    model = Net(n_feature = checkpoint['n_feature'], n_hidden = checkpoint['n_hidden'],
                num_hidden_layers = checkpoint['num_hidden_layers'],
                n_component = checkpoint['n_component'])
    
   # model = AttentionLocalPoly(dim_input=checkpoint['dim_input'], dim_model=checkpoint['dim_model'],
   #                           dim_ff=checkpoint['dim_ff'], num_head=checkpoint['num_head'],
   #                          layer_num=checkpoint['layer_num'])
    model.load_state_dict(checkpoint['state_dict'])
    _ = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay = 0.0001)
    optimizer.load_state_dict(checkpoint['optimizer'])
    best_epoch = checkpoint['epoch']
    return model, optimizer, best_epoch
