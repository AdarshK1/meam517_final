import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import wandb
from model import Net
from dataloader import TrajDataset
from torch.utils.data import DataLoader
import time
import sys
from datetime import datetime
import os

# eventually we can do sweeps with this setup
hyperparameter_defaults = dict(
    batch_size=630,
    learning_rate=0.0001,
    weight_decay=0.00005,
    epochs=100,
    test_iters=50,
    num_workers=16,
    with_x=False,
    x_dim=0,
    u_dim=3,
    fcn_1=800,
    fcn_2=400,
    fcn_3=200,
    u_max=75,
)

dt = datetime.now().strftime("%m_%d_%H_%M")
name_str = "_only_u_normed"
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
                       weight_decay=config.weight_decay, amsgrad=False)

criterion = nn.L1Loss()

sample_fname = "/home/adarsh/software/meam517_final/data_v2/"
dset = TrajDataset(sample_fname, with_x=config.with_x, max_u=config.u_max)

train_loader = DataLoader(dset, batch_size=config.batch_size, num_workers=config.num_workers)

for epoch in range(config.epochs):
    for i_batch, sample_batched in enumerate(train_loader):
        t1 = time.time()
        input, output = sample_batched

        optimizer.zero_grad()  # zero the gradient buffers

        input = input.float().cuda()

        output = output.float()

        # forward! doesn't use semantic rn but we have it i guess
        output_predicted = net(input)

        loss = criterion(output_predicted.cpu().float(), output)

        wandb.log({'epoch': epoch, 'iteration': i_batch, 'loss': loss.item()})
        print({'epoch': epoch, 'iteration': i_batch, 'loss': loss.item()})

        # backprop
        loss.backward()
        optimizer.step()  # Does the update

        backup_path = backup_dir + "/model.ckpt"

        torch.save(net.state_dict(), backup_path)
        t2 = time.time()
