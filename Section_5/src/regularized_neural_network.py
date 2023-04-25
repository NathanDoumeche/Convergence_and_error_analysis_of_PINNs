import torch
from torch import nn
from torch.autograd import Variable


class NeuralNetwork(nn.Module):
    def __init__(self, D):
        super(NeuralNetwork, self).__init__()
        self.dense = nn.Sequential(nn.Linear(2, D), nn.Tanh(), nn.Linear(D, D), nn.Tanh(), nn.Linear(D, 1))

    def forward(self, x):
        xGrad = Variable(x, requires_grad=True)
        r = self.dense(xGrad)

        # Gradient computations
        gradientR = torch.autograd.grad(r, xGrad, torch.ones_like(r), create_graph=True)
        self.gradientR = gradientR[0]
        self.gradient2R1 = \
        torch.autograd.grad(gradientR[0][:, 0], xGrad, torch.ones_like(gradientR[0][:, 0]), create_graph=True)[0]
        self.gradient2R2 = \
        torch.autograd.grad(gradientR[0][:, 1], xGrad, torch.ones_like(gradientR[0][:, 1]), create_graph=True)[0]
        return r


def mean_squared(vector):
    return nn.functional.mse_loss(vector, torch.zeros(vector.size()))


def advection_constraint(model, n_r):
    return mean_squared(model.gradientR[0:n_r, 0] + model.gradientR[0:n_r, 1])


def boundary_initial_conditions(pred, n_r, n_e, h):
    return nn.functional.mse_loss(pred[n_r:n_r + n_e, ], h)


def sobolev_regularization(model, n_r, pred):
    sobolev0 = pred[0:n_r, ]
    sobolev1 = model.gradientR[0:n_r, ]
    sobolev2_1 = model.gradient2R1[0:n_r, ]
    sobolev2_2 = model.gradient2R2[0:n_r, ]
    return mean_squared(sobolev0) + mean_squared(sobolev1) + mean_squared(sobolev2_1) + mean_squared(sobolev2_2)


def lossGrad(model, x_r, x_e, h, x, y):
    x_tot = torch.cat((x_r, x_e, x))
    pred = model(x_tot)

    data_loss = 0 if len(x) == 0 else nn.functional.mse_loss(pred[len(x_r) + len(x_e):], y)
    conditions_loss = 0 if len(x_e) == 0 else boundary_initial_conditions(pred, len(x_r), len(x_e), h)
    differential_loss = 0 if len(x_r) == 0 else advection_constraint(model, len(x_r))
    sobolev_loss = 0 if len(x_r) == 0 else sobolev_regularization(model, len(x_r), pred)
    return data_loss, conditions_loss, differential_loss, sobolev_loss
