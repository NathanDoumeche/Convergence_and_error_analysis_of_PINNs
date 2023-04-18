import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import os as os

from src.figures import figure_monitoring, figure_nn, linear_reg, PI
from src.regularized_neural_network import NeuralNetwork, lossGrad
from src.training_validation import point_sampling, train

torch.manual_seed(42)


def u_star(x):
    return torch.exp(x[:, 0] - x[:, 1]) + 0.1 * torch.cos(2*np.pi*x[:, 1])


def initial_condition(x):
    return torch.exp(-x)


def boundary_condition(t):
    return torch.exp(t)


def simulation(n, n_e, n_r, n_val, D, weight_decay_exponent):
    noise = 0.1

    # Hyperparameters
    epochs = 1000
    batch_number = 1
    lr = 10**(-3)

    weight_decay = n_r**weight_decay_exponent
    lambda_t = 0.1/np.log(n)
    lambda_d = 10*n**(1/2)

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

    train_loss, test_loss, overfitting_gap_list, physics_inconsistency_list = [], [], [], []

    for t in range(epochs):
        print(f"Epoch p = {t+1}")
        loss_train, loss_test, overfitting_gap, physics_inconsistency = \
            train(train_loader_r, train_loader_e, train_loader_data, val_Xr, val_Xe, val_data, model, loss_fn,
                  optimizer, lambda_t, lambda_d)
        train_loss.append(loss_train)
        test_loss.append(loss_test)
        overfitting_gap_list.append(overfitting_gap)
        physics_inconsistency_list.append(physics_inconsistency)

    figure_monitoring(train_loss, test_loss, overfitting_gap_list, physics_inconsistency_list, n, n_e, n_r, D)
    torch.save(model.state_dict(), os.path.join("Outputs", "NN_weights", "model_"+str(n)+".pt"))
    return np.log(np.array(test_loss))[-1], np.log(np.array(physics_inconsistency_list))[-1]


if __name__ == "__main__":
    list_n, test_loss, physics_inconsistency = [], [], []
    n_e = 10**4
    n_r = n_e
    n_val = 10**2
    D = 100
    weight_decay_exponent = -1/2

    for order_of_magnitude in np.arange(1, 3+1/8, 1/4):
        n = int(10**order_of_magnitude)
        print("Training model for n = "+str(n))
        list_n.append(np.log(n))
        test_loss_n, physics_inconsistency_n = simulation(n, n_e, n_r, n_val, D, weight_decay_exponent)
        test_loss.append(test_loss_n)
        physics_inconsistency.append(physics_inconsistency_n)

    pd.DataFrame({"log(n)": list_n, "log(Validation_loss)": test_loss, "log(PI)": physics_inconsistency}).to_csv(os.path.join("Outputs", "validation_loss.csv"))
    linear_reg()
    PI()
    figure_nn(u_star, D, 10**3)
