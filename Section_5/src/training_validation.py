import numpy as np
import torch


def point_sampling(u_star, initial_condition, boundary_condition, n, n_e, n_r, noise):
    #Data point sampling
    train_t = (torch.rand(n, 1) / 2).reshape(-1, 1)
    train_x = torch.rand(n, 1).reshape(-1, 1)
    train_X = torch.concat((train_t, train_x), dim=1)
    train_y = (u_star(train_X)).reshape(-1, 1) + noise * torch.normal(mean=torch.zeros(n)).reshape(-1, 1)
    train_data = torch.cat((train_X, train_y), dim=1)

    #Intial / Boundary conditions point sampling
    initial_points = torch.cat((torch.zeros((n_e // 2, 1)), torch.rand(n_e // 2, 1)), dim=1)
    initial_values = (initial_condition(initial_points[:, 1])).reshape(-1, 1)
    initial_conditions = torch.cat((initial_points, initial_values), dim=1)

    boundary_points = torch.cat((torch.rand(n_e // 2, 1), torch.zeros((n_e // 2, 1))), dim=1)
    boundary_values = (boundary_condition(boundary_points[:, 0])).reshape(-1, 1)
    boundary_conditions = torch.cat((boundary_points, boundary_values), dim=1)
    train_Xe = torch.cat((initial_conditions, boundary_conditions))

    #Collocation point sampling
    train_Xr = torch.rand(n_r, 2)
    return train_data, train_Xe, train_Xr


def train(train_loader_r, train_loader_e, train_loader_data, val_r, val_e, val_data, model, loss_fn, optimizer,
          lambda_t, lambda_d):
    #Training
    batch_number = len(train_loader_r)

    model.train()
    loss_train_Tot, empirical_PI = [], []

    train_r = iter(train_loader_r)
    train_e = iter(train_loader_e)
    train_XY = iter(train_loader_data)

    for batch in range(batch_number):
        x_r, XeH, XY = next(train_r), next(train_e), next(train_XY)
        x_e, h = XeH[:, 0:2].reshape(-1, 2), XeH[:, 2].reshape(-1, 1)
        x, y = XY[:, 0:2].reshape(-1, 2), XY[:, 2].reshape(-1, 1)

        # Backpropagation
        optimizer.zero_grad()
        loss = loss_fn(model, x_r, x_e, h, x, y)
        lossTot = lambda_d * loss[0] + loss[1] + loss[2] + lambda_t * loss[3]
        lossTot.backward(retain_graph=True)
        optimizer.step()

        loss_train_Tot.append(lossTot.item())
        empirical_PI.append((loss[1] + loss[2] + lambda_t * loss[3]).item())
    loss_mean_train = np.mean(np.array(loss_train_Tot))
    empirical_PI_mean = np.mean(np.array(empirical_PI))
    
    #Validation : new X_r and X_e
    model.eval()
    loss_val_tot = []

    val_Xe, val_h = val_e[:, 0:2].reshape(-1, 2), val_e[:, 2].reshape(-1, 1)
    loss = loss_fn(model, x_r=val_r, x_e=val_Xe, h=val_h, x=torch.empty(0), y=torch.empty(0))
    loss_val_tot.append((loss[1] + loss[2] + lambda_t * loss[3]).item())
    overfitting_gap = empirical_PI_mean - np.mean(np.array(loss_val_tot))
    physics_inconsistency = np.mean(np.array((loss[1] + loss[2]).item()))

    #Test : new (X,Y)
    val_x, val_y = val_data[:, 0:2].reshape(-1, 2), val_data[:, 2].reshape(-1, 1)
    loss = loss_fn(model, x_r=torch.empty(0), x_e=torch.empty(0), h=torch.empty(0), x=val_x, y=val_y)
    loss_mean_test = np.mean(np.array(loss[0].item()))

    return loss_mean_train, loss_mean_test, overfitting_gap, physics_inconsistency
