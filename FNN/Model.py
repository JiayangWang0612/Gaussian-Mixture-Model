import torch
import torch.nn as nn

class Net(torch.nn.Module):
    def __init__(self,n_feature,n_hidden, num_hidden_layers, n_component):
        super(Net,self).__init__()
        self.hidden = torch.nn.Linear(n_feature,n_hidden)
        self.relu = nn.ReLU()
        self.hidden_layers = nn.ModuleList()
        for i in range(num_hidden_layers - 1):
            self.hidden_layers.append(nn.Linear(n_hidden, n_hidden))
        self.predict = torch.nn.Linear(n_hidden,n_component)

        
    def forward(self,x):
        out_y = self.hidden(x)
        out_y = self.relu(out_y)
        for layer in self.hidden_layers:
            out_y = self.relu(layer(out_y))        
        out_y = self.predict(out_y)
        return out_y