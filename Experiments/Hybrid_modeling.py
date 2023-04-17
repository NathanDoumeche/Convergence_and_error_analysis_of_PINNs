import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import os as os

from src.figures import figure_monitoring, linear_reg
from src.regularized_neural_network import NeuralNetwork, lossGrad
from src.training_validation import point_sampling, train

torch.manual_seed(42)


def u_star(x):
    return torch.exp(x[:, 0] - x[:, 1]) + 0.1 * torch.cos(x[:, 1])

def initial_condition(x):
    return torch.exp(-x)

def boundary_condition(t):
    return torch.exp(t)

def simulation(n, n_e, n_r, n_val, D):
    noise = 0.1

    # Hyperparameters
    epochs = 1000
    batch_number = 1
    lr = 10**(-3)

    weight_decay = n_r**(-2)
    lambda_t = 0.1/np.log(n)
    lambda_d = n**(1/2)

    # Learning and validation dataset: 0 < t,x < 1
    train_data, train_Xe, train_Xr = point_sampling(u_star, initial_condition, boundary_condition, n, n_e, n_r, noise)
    val_data, val_Xe, val_Xr = point_sampling(u_star, initial_condition, boundary_condition, n=n_val, n_e=n_val, n_r=n_val, noise=0)

    # Create data loaders
    train_loader_r = DataLoader(train_Xr, batch_size=n_r//batch_number)
    train_loader_e = DataLoader(train_Xe, batch_size=n_e//batch_number)
    train_loader_data = DataLoader(train_data, batch_size=n//batch_number)

    model = NeuralNetwork(D)
    loss_fn = lossGrad
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loss, val_loss, overfitting_gap_list = [], [], []

    for t in range(epochs):
        print(f"Epoch p = {t+1}")
        loss_train, loss_val, overfitting_gap = train(train_loader_r, train_loader_e, train_loader_data, val_Xr,
                                                      val_Xe, val_data, model, loss_fn, optimizer, lambda_t, lambda_d)
        train_loss.append(loss_train)
        val_loss.append(loss_val)
        overfitting_gap_list.append(overfitting_gap)

    figure_monitoring(train_loss, val_loss, overfitting_gap_list, n, n_e, n_r, D)
    return np.log(np.array(val_loss))[-1]


if __name__ == "__main__":
    list_n, val_loss=[], []
    n_e = int(10**2.5)
    n_r = n_e
    n_val = 10**2
    D=300

    for order_of_magnitude in np.arange(1, 3+1/4, 1/2):
        n = int(10**order_of_magnitude)
        print("Training model for n = "+str(n))
        list_n.append(np.log(n))
        val_loss.append(simulation(n, n_e, n_r, n_val, D))

    pd.DataFrame({"log(n)": list_n, "log(Validation_loss)": val_loss}).to_csv(os.path.join("Outputs_PINNs", "validation_loss.csv"))
    linear_reg()
