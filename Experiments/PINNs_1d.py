import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from os.path import join
import os as os
from torch.autograd import Variable


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Linear(1, 100),
            nn.Tanh(),
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.Linear(100, 1), )

    def forward(self, x):
        xGrad = Variable(x, requires_grad=True)
        r = self.conv(xGrad)
        self.gradientR = torch.autograd.grad(r, xGrad, torch.ones_like(r), create_graph=True)
        self.gradient2R = torch.autograd.grad(self.gradientR, xGrad, torch.ones_like(r), create_graph=True)
        return r


def lossGrad(pred, target, constraint):
    gradient2R = nn.functional.relu(-constraint[0])
    return 100 * nn.functional.mse_loss(gradient2R, torch.zeros(gradient2R.size())) + nn.functional.mse_loss(pred, target)


def train(train_dataloader, val_dataloader, model, loss_fn, optimizer):
    model.train()

    loss_train = []
    for batch, X in enumerate(train_dataloader):
        x = X[:, 0].reshape(-1, 1)
        y = X[:, 1].reshape(-1, 1)

        # Compute prediction error
        pred = model(x)
        loss = loss_fn(pred, y, model.gradient2R)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        loss_train.append(loss.item())

    loss_mean_train = np.mean(np.array(loss_train))
    print(f"loss: {loss_mean_train}\n")

    model.eval()

    print("Validation")

    lossVal = []

    for batch, X in enumerate(val_dataloader):
        x = X[:, 0].reshape(-1, 1)
        y = X[:, 1].reshape(-1, 1)
        # Compute prediction error
        pred = model(x)
        loss = loss_fn(pred, y, model.gradient2R)
        lossVal.append(loss.item())

    loss_mean_val = np.mean(np.array(lossVal))
    print(f"Validation loss: {loss_mean_val:>7f}")

    return loss_mean_train, loss_mean_val


if __name__ == '__main__':
    # HYPERPARAMETERS

    epochs = 50
    batch_size = 20
    lr = 1e-3
    weight_decay = 0
    model_name = "model_deep"

    # LOAD DATASET

    # n = m
    n = 1000

    # Apprentissage1 : 0 < x < 1
    train_x = torch.rand(n, 1)
    train_y = train_x ** 2 + torch.cos(30 * train_x) / 10 + torch.randn(n, 1) / 10
    train_dataset = torch.cat((train_x, train_y), dim=1)

    # Validation
    val_x = torch.rand(100, 1)
    val_y = val_x ** 2 + torch.cos(30 * val_x) / 10
    val_dataset = torch.cat((val_x, val_y), dim=1)

    # Create data loaders.
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    model = NeuralNetwork()
    # print(model)

    loss_fn = lossGrad
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loss = []
    val_loss = []
    if not os.path.exists('NN_weights'):
        os.makedirs('NN_weights')
    for t in range(epochs):
        print(f"\nEpoch {t + 1}\n-------------------------------")
        loss_train, loss_val = train(train_dataloader, val_dataloader, model, loss_fn, optimizer)
        if len(val_loss) > 0 and loss_val < np.min(val_loss):
            torch.save(model.state_dict(), join("NN_weights", f"{model_name}.pth"))
            print(f"Saved PyTorch Model State to NN_weights/{model_name}.pth")
        train_loss.append(loss_train)
        val_loss.append(loss_val)

    print("Done!")

    plt.figure()
    plt.plot(np.array(train_loss), label="train loss")
    plt.plot(np.array(val_loss), label="val loss")
    plt.legend()
    plt.xlabel("Epochs")
    plt.savefig(os.path.join("Outputs", "perf.pdf"), bbox_inches="tight")

    plt.figure()
    plt.scatter(train_x.numpy(), train_y.numpy(), label="Entraînement avec du bruit")
    plt.legend()
    plt.xlabel("Epochs")
    plt.savefig(os.path.join("Outputs", "data.pdf"), bbox_inches="tight")
