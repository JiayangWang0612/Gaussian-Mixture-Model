import torch
import torch.nn as nn
from Models import *
from torch.optim import Adam
import numpy as np


#def loss_function(y, y_hat, weight, mean, variance):
#    m = len(y_hat)
#    n = len(mean.t())
#    a = torch.empty(m,n)
#    b = torch.empty(m,n)
#    for t in range(m):
#        for i in range(n):
#            b[t,i] = torch.exp( - torch.square(y[t] - y_hat[t] - mean[t-1,:][i]) / 2 / variance[t-1,:][i])
#            a[t,i] = weight[t-1,:][i] * 1 / (torch.sqrt(torch.tensor([2*3.14]))
#                                      * torch.square(variance[t-1,:][i])) * b[t,i]
#    a_t = torch.sum(a, dim = 1)
#    loss = -torch.sum(torch.log(a_t))
#    print(loss)
#    return loss

class LikelihoodLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, y, y_hat, weight, mean, variance):
        m = len(y_hat)
        n = len(mean.t())
        a = torch.empty(m,n)
        b = torch.empty(m,n)
        for t in range(m):
            for i in range(n):
#                b[t,i] = torch.exp( - torch.square(y[t] - y_hat[t] - mean[t,:][i]) / 2 / variance[t,:][i])
#                print("b:", b)
                a[t,i] = weight[t,:][i] * 1 / torch.sqrt(torch.tensor([2*3.14])) / torch.sqrt(variance[t,:][i]) * torch.exp( - torch.square(y[t] - y_hat[t] - mean[t,:][i]) / 2 / variance[t,:][i])
        a_t = torch.sum(a, dim = 1)
        loss = -torch.mean(torch.log(a_t + 0.000001))
#        print(loss)
        return loss        


class LikelihoodLoss2(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, y, y_hat, weight, mean, variance):
        size = len(y)
        m = len(y_hat)
        n = len(mean.t())
#        a = torch.empty(m,n)
#        b = torch.empty(m,n)
#        c = torch.empty(m,n)
#        d = torch.empty(m,n)
        y = y.reshape(size)
        y_hat = y_hat.reshape(size)
        y_change = torch.stack((y,y,y),1)
#        print(y_change.size())
        y_hatchange = torch.stack((y_hat,y_hat,y_hat),1)
        b = y_change - y_hatchange - mean
#        print(b.size())
        c = torch.exp(-b.mul(b) / 2 / variance)
        d = weight / torch.sqrt(variance)
        a = 1 / torch.sqrt(torch.tensor([2*3.14])) * d.mul(c)
#        a = 1 / torch.sqrt(torch.tensor([2*3.14])) * (weight / torch.sqrt(variance)).mul(torch.exp(-(y_change - y_hatchange - mean).mul(y_change - y_hatchange - mean) / 2 / variance))
#        print(a)
        a_t = torch.sum(a, dim = 1)
#        print(a_t)
        loss = -torch.mean(torch.log(a_t))
#        print(loss)
        return loss               

def gong_function(x):
    y = (1/x) * np.sin(15/x)
    return y


def save_model(folder, k, n_feature, n_hidden, n_component,
               model, optimizer, num_hidden_layers, best=False):
    checkpoint = {'n_feature': n_feature,
                  'n_hidden': n_hidden,
                  'n_component': n_component,
                  'state_dict': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'num_hidden_layers': num_hidden_layers,
                  'epoch': k}
    if best:
        k = "best"
    torch.save(checkpoint, folder + str(k) + '_' + 'checkpoint.pth')


def load_model(file_path, device):
    checkpoint = torch.load(file_path)
    model = GaussianMixture(n_feature = checkpoint['n_feature'], n_hidden = checkpoint['n_hidden'],
                n_component = checkpoint['n_component'], 
                num_hidden_layers = checkpoint['num_hidden_layers'], dropout_prob=0.1)
    
   # model = AttentionLocalPoly(dim_input=checkpoint['dim_input'], dim_model=checkpoint['dim_model'],
   #                           dim_ff=checkpoint['dim_ff'], num_head=checkpoint['num_head'],
   #                          layer_num=checkpoint['layer_num'])
    model.load_state_dict(checkpoint['state_dict'])
    model.dropout1.requires_grad = True
    _ = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    optimizer.load_state_dict(checkpoint['optimizer'])
    best_epoch = checkpoint['epoch']
    return model, optimizer, best_epoch