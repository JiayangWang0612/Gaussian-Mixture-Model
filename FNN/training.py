import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
from Model import *
from pathlib import Path
from utils import *
from Make_Prediction import *
import csv
import sys
import numpy
import random

device = torch.device("cpu")

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(0)


size = 1

# Define the parameters of the three Gaussian distributions
mean1, std1 = torch.tensor([0.0]), 1.0
mean2, std2 = torch.tensor([0.5]), 0.2
mean3, std3 = torch.tensor([-0.5]), 0.5

# Assign weights to each Gaussian distribution
weight1, weight2, weight3 = 0.3, 0.4, 0.3

# Generate a random sample of data points which follow Gaussian Mixture Model with 3 components
num_samples = 500
samples = []
for i in range(num_samples):
    rand_num = torch.rand(1)
    if rand_num < weight1:
        samples.append(torch.normal(mean1, std1))
    elif rand_num < weight1 + weight2:
        samples.append(torch.normal(mean2, std2))
    else:
        samples.append(torch.normal(mean3, std3))

samples = torch.stack(samples)



## Simulation: y = x + x^2; generate data 
## Cross-validation: divided into 3 parts: train; valid; test.
train_x = torch.unsqueeze(torch.linspace(-2,2,500), 1)
print(len(train_x))
train_y =  train_x + train_x**2 + samples

valid_x = torch.unsqueeze(torch.linspace(-2,2,500), 1)
valid_y = valid_x + valid_x**2 + samples


test_x = torch.unsqueeze(torch.linspace(-2,2,500), 1)
test_y = test_x + test_x**2




# x,y = Variable(train_x), Variable(train_y)

x,y = train_x, train_y
net = Net(1,10,5,1)
n_feature = 1
n_hidden = 10
num_hidden_layers = 5
n_component = 1
print(net)

opt = torch.optim.Adam(net.parameters(),lr = 0.01, weight_decay = 0.0001)

lossfunc = nn.MSELoss()

model_name = "GassianMixture" + "/"
folder = "./Train/" + model_name 
print(folder)
Path(folder).mkdir(parents=True, exist_ok=True)
best_epoch = 0
early_stop_indicator = 0
min_valid_loss = sys.maxsize
plot_loss = []
plot_loss_validation = []


for t in range(100):    # change to 5000 epoches
    save_model(folder, t, n_feature, n_hidden, num_hidden_layers, n_component,
               net, opt)
    net.train()
    out_y = net(x)
    loss = lossfunc(y,out_y)
#    print("epoch is", t, " train_mse:", loss)
    plot_loss.append(loss.item())    
    opt.zero_grad()
    loss.backward()
    opt.step()  
    
    with torch.no_grad():
         net.eval()
         x,y = valid_x, valid_y
         out_y = net(x)
         valid_loss = lossfunc(y, out_y)
#         print("valid_mse:", valid_loss)
#         valid_loss = fakeloss(y,out_y)
         # print("valid - check out: ", check_tensor([valid_loss]))
         loss_valid = valid_loss.item()
         plot_loss_validation.append(loss_valid)

    if loss_valid < min_valid_loss:
       save_model(folder, t, n_feature, n_hidden,num_hidden_layers, n_component,
                  net, opt, best=True)
       min_valid_loss = loss_valid
     # early_stop_indicator = 0
       best_epoch = t
     # else:
     # early_stop_indicator += 1



ck = folder + "best_checkpoint.pth"
# load the best model
model, optimizer, best_epoch = load_model(ck, device)
output = model(train_x)
#print(output)

#for name, param in net.named_parameters():
#    if param.requires_grad:
#        print(name, param.data)



# Plot the training and validation loss
plt.plot(plot_loss, label='Training Loss')
plt.plot(plot_loss_validation, label='Validation Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.show()   



# Make a prediction for testing data
y_pred, y_true = make_prediction((test_x, valid_x, train_x), (test_y, valid_y, train_y), 
                                     device = device, folder=folder)   
print("Important: train_mse:", np.mean(np.square(y_pred - y_true)))
train_mse = np.mean(np.square(y_pred - y_true))



plt.figure(figsize=(16, 8))
plt.scatter(test_x.detach().cpu().numpy(), test_y.detach().cpu().numpy(), label = "Origin Testing Data")
    
y_pred, y_true = make_prediction((train_x, valid_x, test_x), (train_y, valid_y, test_y), 
                                     device = device, folder=folder)
#print("test_mse:", np.mean(np.square(y_pred - y_true)))

plt.scatter(test_x.detach().cpu().numpy(), y_pred, marker="x", color="purple", label = "Prediction of Testing Data")
print(best_epoch)
results = "Plots_" + model_name + "/"
Path(results).mkdir(parents=True, exist_ok=True)
plt.title("Feedfoward_Model")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend(fontsize = 20)
plt.show()
#plt.savefig(results + str(t) + ".png",dpi=800)




# Make a prediction for training data
plt.figure(figsize=(16, 8))
plt.scatter(train_x.detach().cpu().numpy(), train_y.detach().cpu().numpy(), label = "Origin Training Data")
y_pred, y_train = make_prediction((test_x, valid_x, train_x), (test_y, valid_y, train_y), 
                                     device = device, folder=folder)

plt.scatter(train_x.detach().cpu().numpy(), y_pred, marker="x", color="purple", label = "Prediction of Training Data")
plt.legend(fontsize = 20)
plt.show()









output_results = [size, best_epoch, train_mse]
print("Important: test_mse:", np.mean(np.square(y_pred - y_true)))
test_mse = np.mean(np.square(y_pred - y_true))
output_results.append(test_mse)

file = open("Size " + str(size) +  "FeedFoward Neural Network Results.csv", mode="a")
writer = csv.writer(file)
writer.writerow(output_results)
file.close()






#output = net.hidden(train_x[0])
#print(output)
#out_y = net.relu(output)
#print(out_y)
#for layer in net.hidden_layers:
#    out_y = net.relu(layer(out_y))  
#    print(out_y)      
#out_y = net.predict(out_y)
#print(out_y)
    
#print(out_y)
#print(torch.var(train_y))










