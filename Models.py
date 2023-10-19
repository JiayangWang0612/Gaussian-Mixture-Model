import torch
import torch.nn as nn

class GaussianMixture(torch.nn.Module):
    def __init__(self,n_feature,n_hidden,n_component, num_hidden_layers, dropout_prob=0.1):
        super(GaussianMixture,self).__init__()       
        self.hidden1 = torch.nn.Linear(n_feature,n_hidden)
        self.dropout1 = torch.nn.Dropout(p=dropout_prob)
        self.hidden2 = torch.nn.Linear(n_feature,n_hidden)
        self.hidden3 = torch.nn.Linear(n_feature,n_hidden)
        self.hidden4 = torch.nn.Linear(n_feature,n_hidden)
        self.relu = nn.ReLU()
        self.hidden_layers = nn.ModuleList()
        for i in range(num_hidden_layers - 1):
            self.hidden_layers.append(nn.Linear(n_hidden, n_hidden))
        self.predict1 = torch.nn.Linear(n_hidden,1)
        self.predict2 = torch.nn.Linear(n_hidden,n_component)
        self.predict3 = torch.nn.Linear(n_hidden,n_component)
        self.predict4 = torch.nn.Linear(n_hidden,n_component)
        
    def forward(self,x):
#       print(x.size())
        out_y = self.hidden1(x)
        out_y = self.relu(out_y)
        for layer in self.hidden_layers:
            out_y = self.relu(layer(out_y))
        out_y = self.dropout1(out_y)
        out_y = self.predict1(out_y)
#        print(out_y)
#        out_y = self.relu(out_y)
#        residue = self.relu(self.hidden1(out_y))
#        out_y = self.predict1(out_y) +residue
        
        out_weight = self.hidden2(x)
        residue = out_weight
        out_weight = self.relu(out_weight)
        for layer in self.hidden_layers:
            out_weight = self.relu(out_weight)
        out_weight = out_weight +residue
        out_weight = self.predict2(out_weight)
        out_weight = torch.softmax(out_weight, dim = 1)
#       print(out_weight.size())        

        out_mean = self.hidden3(x)
        out_mean = self.relu(out_mean)
        for layer in self.hidden_layers:
            out_mean = self.relu(out_mean)
        out_mean = self.predict3(out_mean)
#       print(out_mean[:-1].size())
        
        mean_last = - torch.sum(out_weight[:,:-1].mul(out_mean[:,:-1]), dim = 1) / out_weight[:,-1]
#       print(mean_last.reshape(100,1).size())
#       print(out_mean[:,:-1].size())
        output_mean = torch.cat((out_mean[:,:-1].t(), mean_last.reshape(len(x),1).t())).t()
        
        out_variance = self.hidden4(x)
        out_variance = self.relu(out_variance)
        for layer in self.hidden_layers:
            out_variance = self.relu(out_variance)
        out_variance = self.predict4(out_variance)
        output_variance = torch.abs(out_variance)
#        print(out_y, out_weight, output_mean, output_variance)
        return out_y, out_weight, output_mean, output_variance
    


