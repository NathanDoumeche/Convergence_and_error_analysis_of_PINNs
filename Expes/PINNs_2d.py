import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
import os as os
from torch.autograd import Variable

torch.manual_seed(42)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.dense = nn.Sequential(nn.Linear(2, 300), nn.Tanh(), nn.Linear(300, 300), nn.Tanh(), nn.Linear(300, 1))

    def forward(self, x):
        xGrad = Variable(x, requires_grad = True)
        r = self.dense(xGrad)

        # Gradient computations
        gradientR = torch.autograd.grad(r, xGrad, torch.ones_like(r), create_graph=True)
        self.gradientR = gradientR[0]
        self.gradient2R1 = torch.autograd.grad(gradientR[0][:,0], xGrad, torch.ones_like(gradientR[0][:,0]), create_graph=True)[0]
        self.gradient2R2 = torch.autograd.grad(gradientR[0][:,1], xGrad, torch.ones_like(gradientR[0][:,1]), create_graph=True)[0]
        return r

def mean_squared(vector):
    return nn.functional.mse_loss(vector, torch.zeros(vector.size()))

def sobolev_regularization(model, n_r, pred):
    sobolev0 = pred[0:n_r,]
    sobolev1 = model.gradientR[0:n_r,]
    sobolev2_1 = model.gradient2R1[0:n_r,]
    sobolev2_2 = model.gradient2R2[0:n_r,]
    return mean_squared(sobolev0)+mean_squared(sobolev1)+mean_squared(sobolev2_1)+mean_squared(sobolev2_2)


def advection_constraint(model, n_r):
    return mean_squared(model.gradientR[0:n_r,0]-model.gradientR[0:n_r,1])


def boundary_initial_conditions(pred, n_r, n_e, h):
    return nn.functional.mse_loss(pred[n_r:n_r+n_e,], h)


def lossGrad(model, x_r, x_e, h, x, y):
    x_tot = torch.cat((x_r, x_e, x))
    pred = model(x_tot)
    return nn.functional.mse_loss(pred[len(x_r)+len(x_e):], y), boundary_initial_conditions(pred, len(x_r), len(x_e), h),\
        advection_constraint(model, len(x_r)), sobolev_regularization(model, len(x_r), pred)


def train(train_loader_r, train_loader_e, train_loader_data, val_loader, model, loss_fn, optimizer, lambda_t, lambda_d):
    batch_number = len(train_loader_r)
    model.train()
    loss_trainAcc = []

    train_r = iter(train_loader_r)
    train_e = iter(train_loader_e)
    train_XY = iter(train_loader_data)

    for batch in range(batch_number):
        x_r, XeH, XY = next(train_r), next(train_e), next(train_XY)
        x_e, h = XY[:,0:2].reshape(-1,2), XY[:,2].reshape(-1,1)
        x, y = XY[:,0:2].reshape(-1,2), XY[:,2].reshape(-1,1)

        # Backpropagation
        optimizer.zero_grad()
        loss = loss_fn(model, x_r, x_e, h, x, y)
        lossTot = lambda_d * loss[0] + loss[1] + loss[2] + lambda_t * loss[3]
        lossTot.backward(retain_graph=True)
        optimizer.step()

        loss_trainAcc.append(loss[0].item())
    loss_mean_train = np.mean(np.array(loss_trainAcc))

    model.eval()
    lossValAcc = []
    for batch, X in enumerate(val_loader):
        x = X[:,0:2].reshape(-1,2)
        y = X[:,2].reshape(-1,1)
        # Compute prediction error
        pred = model(x)
        lossValAcc = nn.functional.mse_loss(pred, y).item()
    loss_mean_val = np.mean(np.array(lossValAcc))
    return loss_mean_train, loss_mean_val


def simulation(n, n_e, n_r, n_val):
    # Hyperparameters
    epochs, batch_number, lr = 50, 10, 10**(-4)

    # PINNs hyperparameter
    weight_decay = n_r**(-1/2)
    lambda_t = 0.1*(np.log(n))**(-1)
    lambda_d = (n**(1/2)) * lambda_t * 10

    #Model
    noise = 0.1

    # Learning dataset: 0 < t,x < 1
    train_Xr = torch.rand(n_r,2)

    initial_points = torch.cat((torch.zeros((n_e//2,1)), torch.rand(n_e//2,1)), dim=1)
    initial_values = (torch.exp(-initial_points[:,1])).reshape(-1,1)
    initial_conditions = torch.cat((initial_points, initial_values), dim=1)

    boundary_points = torch.cat((torch.rand(n_e//2,1), torch.zeros((n_e//2,1))), dim=1)
    boundary_values = (torch.exp(boundary_points[:,0])).reshape(-1,1)
    boundary_conditions = torch.cat((boundary_points, boundary_values), dim=1)
    train_Xe = torch.cat((initial_conditions, boundary_conditions))

    train_X = torch.rand(n, 2)
    train_y = (torch.exp(train_X[:,0]-train_X[:,1])).reshape(-1,1) + noise*torch.rand(n,1)
    train_data = torch.cat((train_X, train_y), dim=1)

    #Validation dataset: 0 < t,x < 1
    val_x = torch.rand(n_val, 2)
    val_y = (torch.exp(val_x[:,0]-val_x[:,1])).reshape(-1,1)
    val_dataset = torch.cat((val_x, val_y), dim=1)

    # Create data loaders
    train_loader_r = DataLoader(train_Xr, batch_size=n_r//batch_number)
    train_loader_e = DataLoader(train_Xe, batch_size=n_e//batch_number)
    train_loader_data = DataLoader(train_data, batch_size=n//batch_number)
    val_loader = DataLoader(val_dataset, batch_size=n_val)

    model = NeuralNetwork()
    loss_fn = lossGrad
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loss = []
    val_loss = []
    for t in range(epochs):
        print(f"Epoch {t+1}")
        loss_train, loss_val = train(train_loader_r, train_loader_e, train_loader_data, val_loader,
                                     model, loss_fn, optimizer, lambda_t, lambda_d)
        train_loss.append(loss_train)
        val_loss.append(loss_val)

    plt.figure()
    plt.plot(np.log(np.array(train_loss)), label="ln(train loss)")
    plt.plot(np.log(np.array(val_loss)), label="ln(val loss)")
    plt.legend()
    plt.xlabel("Epochs")
    plt.savefig(os.path.join("Outputs_PINNs_2d", "perf_"+str(n)+".pdf"), bbox_inches="tight")

    return np.log(np.array(val_loss))[-1]


if __name__ == "__main__":
    list_n, val_loss=[], []
    for order_of_magnitude in np.arange(1, 5, 0.5):
        n, n_e, n_val = int(10**order_of_magnitude), 10**5, 10**5
        n_r = n_e

        print("Training model for n = "+str(n))
        val_loss.append(simulation(n, n_e, n_r, n_val))
        list_n.append(np.log(n))
    pd.DataFrame({"log(n)": list_n, "log(Validation_loss)": val_loss}).to_csv(os.path.join("Outputs_PINNs_2d", "validation_loss.csv"))
