import torch
import numpy as np
from utils import *
import pandas as pd
from sklearn.metrics import roc_auc_score


def make_prediction(X, Y, device=torch.device("cpu"), folder="./Train/", data_type="test"):
    compute_loss = torch.nn.MSELoss()
    ck = folder + "best_checkpoint.pth"
    # load the best model
    model, optimizer, best_epoch = load_model(ck, device)
    train_X, valid_X, test_X = X
    train_Y, valid_Y, test_Y = Y

    with torch.no_grad():
        model.eval()
        x, y = test_X.to(device), test_Y.to(device)
        test_output, out_weight, out_mean, out_variance = model(x)
        test_loss = compute_loss(test_output, y)
        loss = test_loss.item()
        y_pred = test_output.detach().cpu().numpy()
        y_true = y.detach().cpu().numpy()

#   print(data_type + " MSE :", loss)

    y_pred = np.array(y_pred).reshape(-1, )
    y_true = np.array(y_true).reshape(-1, )
    

    return y_pred, y_true

