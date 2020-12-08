import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import wandb
from model import Net, FeasibilityNet
from dataloader import TrajDataset
from torch.utils.data import DataLoader
import time
import sys
from datetime import datetime
import os


def train_u_net():
    # eventually we can do sweeps with this setup
    hyperparameter_defaults = dict(
        batch_size=721,
        learning_rate=0.001,
        weight_decay=0.001,
        epochs=300,
        test_iters=50,
        num_workers=16,
        with_x=False,
        x_dim=0,
        u_dim=3,
        fcn_1=250,
        fcn_2=120,
        fcn_3=50,
        u_max=np.array([25, 25, 10])
    )

    dt = datetime.now().strftime("%m_%d_%H_%M")
    name_str = "_sum_reduction"

    wandb.init(project="517_final", config=hyperparameter_defaults, name=dt + name_str)
    config = wandb.config

    backup_dir = "models/" + dt + name_str

    os.makedirs(backup_dir, exist_ok=True)
    net = Net(x_dim=config.x_dim,
              u_dim=config.u_dim,
              fcn_size_1=config.fcn_1,
              fcn_size_2=config.fcn_2,
              fcn_size_3=config.fcn_3).cuda().float()

    # the usual suspects
    optimizer = optim.Adam(net.parameters(), lr=config.learning_rate, betas=(0.9, 0.999), eps=1e-8,
                           weight_decay=config.weight_decay, amsgrad=True)

    criterion = nn.L1Loss(reduction="sum")

    sample_fname = "/home/adarsh/software/meam517_final/data_v2/"
    dset = TrajDataset(sample_fname, with_x=config.with_x, max_u=config.u_max)

    train_loader = DataLoader(dset, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=True)

    for epoch in range(config.epochs):
        for i_batch, sample_batched in enumerate(train_loader):
            t1 = time.time()
            input, u1_out, u2_out, u3_out = sample_batched

            optimizer.zero_grad()  # zero the gradient buffers

            input = input.float().cuda()

            u1_out = u1_out.float()
            u2_out = u2_out.float()
            u3_out = u3_out.float()

            # forward!
            u1_pred, u2_pred, u3_pred = net(input)

            loss = criterion(u1_pred.cpu().float(), u1_out) + \
                   criterion(u2_pred.cpu().float(), u2_out) + \
                   criterion(u3_pred.cpu().float(), u3_out)

            wandb.log({'epoch': epoch, 'iteration': i_batch, 'loss': loss.item()})
            print({'epoch': epoch, 'iteration': i_batch, 'loss': loss.item()})

            # if i_batch == 0 and epoch % 25 == 0 and epoch > 0:
            # rand_idx = int(np.random.random() * config.batch_size)
            # print("output_gt", output)
            # print("output_pred", output_predicted.cpu().float())

            # backprop
            loss.backward()
            optimizer.step()  # Does the update

            backup_path = backup_dir + "/model.ckpt"

            torch.save(net.state_dict(), backup_path)
            t2 = time.time()


def train_feasibility_classifier():
    # eventually we can do sweeps with this setup
    hyperparameter_defaults = dict(
        batch_size=1155,
        learning_rate=0.001,
        weight_decay=0.001,
        epochs=600,
        test_iters=50,
        num_workers=16,
        with_x=False,
        x_dim=0,
        u_dim=3,
        fcn_1=250,
        fcn_2=120,
        fcn_3=60,
        fcn_4=30,
        fcn_5=10,
        u_max=np.array([25, 25, 10])
    )

    dt = datetime.now().strftime("%m_%d_%H_%M")
    name_str = "_feasibility_classifier"

    wandb.init(project="517_final", config=hyperparameter_defaults, name=dt + name_str)
    config = wandb.config

    backup_dir = "models/" + dt + name_str

    os.makedirs(backup_dir, exist_ok=True)

    net = FeasibilityNet(fcn_size_1=config.fcn_1,
                         fcn_size_2=config.fcn_2,
                         fcn_size_3=config.fcn_3,
                         fcn_size_4=config.fcn_4,
                         fcn_size_5=config.fcn_5,
                         ).cuda().float()

    # the usual suspects
    optimizer = optim.Adam(net.parameters(), lr=config.learning_rate, betas=(0.9, 0.999), eps=1e-8,
                           weight_decay=config.weight_decay, amsgrad=False)

    criterion = nn.BCELoss()

    sample_fname = "/home/adarsh/software/meam517_final/data_v2/"
    dset = TrajDataset(sample_fname, with_x=config.with_x, max_u=config.u_max, feasibility_classifier=True)

    train_loader = DataLoader(dset, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=True)
    for epoch in range(config.epochs):
        for i_batch, sample_batched in enumerate(train_loader):
            t1 = time.time()
            input, output = sample_batched

            optimizer.zero_grad()  # zero the gradient buffers

            input = input.float().cuda()

            # forward!
            pred = net(input)

            loss = criterion(pred.cpu().float(), output.float())

            wandb.log({'epoch': epoch, 'iteration': i_batch, 'loss': loss.item()})
            print({'epoch': epoch, 'iteration': i_batch, 'loss': loss.item()})

            # backprop
            loss.backward()
            optimizer.step()  # Does the update

            backup_path = backup_dir + "/model.ckpt"

            torch.save(net.state_dict(), backup_path)
            t2 = time.time()


if __name__ == "__main__":
    # train_u_net()
    train_feasibility_classifier()
