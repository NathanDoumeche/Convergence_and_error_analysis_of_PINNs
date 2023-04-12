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
        gradientR = torch.autograd.grad(r, xGrad, torch.ones_like(r), create_graph=True)
        self.gradientR = gradientR[0]
        self.gradient2R1 = torch.autograd.grad(gradientR[0][:,0], xGrad, torch.ones_like(gradientR[0][:,0]), create_graph=True)[0]
        self.gradient2R2 = torch.autograd.grad(gradientR[0][:,1], xGrad, torch.ones_like(gradientR[0][:,1]), create_graph=True)[0]
        return r

def mse(y):
    return nn.functional.mse_loss(y, torch.zeros(y.size()))

def lossGrad(pred, target, gradientNN, gradient2_1NN, gradient2_2NN, lambda_t, lambda_d):
    constraint_advection = gradientNN[:,0]-gradientNN[:,1]
    sobolev0 = pred
    sobolev1 = gradientNN
    sobolev2_1 = gradient2_1NN
    sobolev2_2 = gradient2_2NN
    return lambda_d*nn.functional.mse_loss(pred, target), \
        mse(constraint_advection), \
        lambda_t * (mse(sobolev0)+mse(sobolev1)+mse(sobolev2_1)+mse(sobolev2_2))


def train(train_dataloader, val_dataloader, model, loss_fn, optimizer, lambda_t, lambda_d):
    model.train()
    loss_trainAcc = []
    loss_trainCstr = []

    for batch, X in enumerate(train_dataloader):
        x = X[:,0:2].reshape(-1,2)
        y = X[:,2].reshape(-1,1)

        # Compute prediction error
        pred = model(x)
        loss = loss_fn(pred, y, model.gradientR, model.gradient2R1, model.gradient2R2, lambda_t, lambda_d)
        lossTot = loss[0]+loss[1]+loss[2]

        # Backpropagation
        optimizer.zero_grad()
        lossTot.backward(retain_graph=True)
        optimizer.step()

        loss_trainAcc.append(loss[0].item()/lambda_d)
        loss_trainCstr.append(loss[1].item())
    loss_mean_train = np.mean(np.array(loss_trainAcc))

    model.eval()
    lossValAcc = []
    for batch, X in enumerate(val_dataloader):
        x = X[:,0:2].reshape(-1,2)
        y = X[:,2].reshape(-1,1)
        # Compute prediction error
        pred = model(x)
        lossValAcc = nn.functional.mse_loss(pred, y).item()
    loss_mean_val = np.mean(np.array(lossValAcc))
    return loss_mean_train, loss_mean_val


def simulation(n):
    # HYPERPARAMETERS
    epochs = 50
    batch_size = n//10
    noise = 0.1
    lr = 1e-4
    model_name = "model_deep_pinn"

    # PINNs hyperparameter
    weight_decay = n**(-1/2)
    lambda_t = 0.1*(np.log(n))**(-1)
    lambda_d = (n**(1/2)) * lambda_t * 10

    # Learning dataset: 0 < t,x < 1
    train_x = torch.rand(n,2)
    train_y = (torch.exp(train_x[:,0]-train_x[:,1])).reshape(-1,1) + noise*torch.rand(n,1)
    train_dataset = torch.cat((train_x, train_y), dim=1)

    #Validation dataset: 0 < t,x < 1
    m = 100000
    val_x = torch.rand(m, 2)
    val_y = (torch.exp(val_x[:,0]-val_x[:,1])).reshape(-1,1)
    val_dataset = torch.cat((val_x, val_y), dim=1)

    # Create data loaders.
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=m)

    model = NeuralNetwork()
    loss_fn = lossGrad
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loss = []
    val_loss = []
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    for t in range(epochs):
        print(f"\nEpoch {t+1}\n-------------------------------")
        loss_train, loss_val = train(train_dataloader, val_dataloader, model, loss_fn, optimizer, lambda_t, lambda_d)
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
        print("Training model for n = "+str(10**order_of_magnitude))
        val_loss.append(simulation(int(10**order_of_magnitude)))
        list_n.append(np.log(int(10**order_of_magnitude)))
    pd.DataFrame({"log(n)": list_n, "log(Validation_loss)": val_loss}).to_csv(os.path.join("Outputs_PINNs_2d", "validation_loss.csv"))
