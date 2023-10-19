import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
from Models import *
from pathlib import Path
from utils import *
from Make_Prediction import *
import csv
import sys
import numpy
import random
from torch.optim.lr_scheduler import StepLR


device = torch.device("cpu")

torch.manual_seed(0) #set.seed()
random.seed(0) #set.seed()
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


## Simulation: y = x^2; generate data 
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
gm = GaussianMixture(1,10,3,5) #Use GaussianMixture Model
# Choose a suitable parameter (no reason for choosing 1;10;;3;5, just do a simulation)
n_feature = 1
n_hidden = 10
n_component = 3
num_hidden_layers = 5
print(gm)

#Choose a suitable learning rate and weight decay
opt = torch.optim.Adam(gm.parameters(),lr = 0.01, weight_decay = 0.0001)
scheduler = StepLR(opt, step_size=1000, gamma=0.8)
lossfunc = LikelihoodLoss2() # Use the second LikelihoodLoss (in utils.py)
mse = nn.MSELoss()

model_name = "GassianMixture" + "/"
folder = "./Train/" + model_name 
print(folder)
Path(folder).mkdir(parents=True, exist_ok=True)
best_epoch = 0
early_stop_indicator = 0
min_valid_loss = sys.maxsize
epoch = 100  # Run 5000 iterations  
plot_loss = []
plot_loss_validation = []



for t in range(epoch):
    save_model(folder, t, n_feature, n_hidden, n_component,
               gm, opt, num_hidden_layers)
    gm.train()
    out_y, out_weight, out_mean, out_variance = gm(x)
#    print("train_mse:", mse(y, out_y))
    loss = lossfunc(y, out_y, out_weight, out_mean, out_variance)
#    print(loss.item())
    plot_loss.append(loss.item())
#    loss = fakeloss(y,out_y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    scheduler.step()
    
    with torch.no_grad():
         gm.eval()
         x,y = valid_x, valid_y
         out_y, weight, mean, variance = gm(x)
#         print("valid_mse:", mse(y, out_y))
         valid_loss = lossfunc(y, out_y, weight, mean, variance)
#         valid_loss = fakeloss(y,out_y)
#          print("valid - check out: ", check_tensor([valid_loss]))
         loss_valid = valid_loss.item()
         plot_loss_validation.append(loss_valid)

    if loss_valid < min_valid_loss:
       save_model(folder, t, n_feature, n_hidden, n_component,
                  gm, opt, num_hidden_layers, best=True)
       min_valid_loss = loss_valid
     # early_stop_indicator = 0
       best_epoch = t
     # else:
     # early_stop_indicator += 1
        
    
# Use function make_prediction (in file Make_Prediction.py)    
    y_pred, y_true = make_prediction((train_x, valid_x, test_x), (train_y, valid_y, test_y), 
                                     device = device, folder=folder)
#    print("test_mse:", np.mean(np.square(y_pred - y_true)))
#    plt.plot(x.detach().cpu().numpy(), y_pred, marker="x", color="purple")
#    print(best_epoch)
#    results = "Plots_" + model_name + "/"
#    Path(results).mkdir(parents=True, exist_ok=True)
#    plt.title("Gaussian_Mixture_Model")
#    plt.xlabel("X")
#    plt.ylabel("Y")
#    plt.show()
#    plt.savefig(results + str(t) + ".png",
#                dpi=800)
   

#for name, param in gm.named_parameters():
#    if param.requires_grad:
#        print(name, param.data)

#output = gm(train_x)

#print(output.item())


ck = folder + "best_checkpoint.pth"
# load the best model
model, optimizer, best_epoch = load_model(ck, device)
output = model(train_x)
#print(output)



# Plot the training and validation loss
plt.plot(plot_loss, label='Training Loss')
plt.plot(plot_loss_validation, label='Validation Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.show()   

#plt.plot(plot_loss_validation, color = "purple")
# Plot training prediction compared with training data
plt.figure(figsize=(16, 8))
plt.scatter(train_x.detach().cpu().numpy(), train_y.detach().cpu().numpy(), label = 'Original Data')

   
y_pred, y_true = make_prediction((test_x, valid_x, train_x), (test_y, valid_y, train_y), 
                                     device = device, folder=folder)

plt.plot(train_x.detach().cpu().numpy(), y_pred, marker="x", color="purple", label = 'Prediction')


print(best_epoch)
results = "Plots_" + model_name + "/"
Path(results).mkdir(parents=True, exist_ok=True)
plt.title("Gaussian_Mixture_Model Training Data Fitting")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend(fontsize = 20)
plt.show()


# Plot test prediction compared with test data
plt.figure(figsize=(16, 8))
plt.scatter(test_x.detach().cpu().numpy(), test_y.detach().cpu().numpy(), label = 'Original Data')
    
y_pred, y_true = make_prediction((train_x, valid_x, test_x), (train_y, valid_y, test_y), 
                                     device = device, folder=folder)

plt.plot(test_x.detach().cpu().numpy(), y_pred, marker="x", color="purple", label = 'Prediction')
plt.title("Gaussian_Mixture_Model Test Data Fitting")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend(fontsize = 20)
plt.show()


# Resave the mse into a csv file
output_results = [size, n_component]
print("test_mse:", np.mean(np.square(y_pred - y_true)))
test_mse = np.mean(np.square(y_pred - y_true))
output_results.append(test_mse)

file = open("Size " + str(size) +  " Gaussian Mixture Model Results.csv", mode="a")
writer = csv.writer(file)
writer.writerow(output_results)
file.close()
