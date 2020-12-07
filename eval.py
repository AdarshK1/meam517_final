import torch
import torch.nn as nn
import numpy as np
from model import Net
from dataloader import TrajDataset
from torch.utils.data import DataLoader
import time
import argparse
import matplotlib.pyplot as plt


def gen_plots(input, preds, gts, u_max=np.array([25, 25, 10])):
    input = input[0, 0, :, :].cpu()
    u1_gt = gts[0][0, :]
    u1_pred = preds[0][0, :].cpu().detach().numpy()
    u2_gt = gts[1][0, :]
    u2_pred = preds[1][0, :].cpu().detach().numpy()
    u3_gt = gts[2][0, :]
    u3_pred = preds[2][0, :].cpu().detach().numpy()

    fig, axs = plt.subplots(4)
    axs[0].plot(u1_gt)
    axs[0].plot(u1_pred)
    axs[0].legend(["u1_gt", "u1_pred"])

    axs[1].plot(u2_gt)
    axs[1].plot(u2_pred)
    axs[1].legend(["u2_gt", "u2_pred"])

    axs[2].plot(u3_gt)
    axs[2].plot(u3_pred)
    axs[2].legend(["u3_gt", "u3_pred"])

    axs[3].imshow(input)

    plt.show()
    time.sleep(1)


parser = argparse.ArgumentParser(description="")
parser.add_argument("--net_path", help="Net checkpoint", required=True)
parser.add_argument("--data_path", help="Data object", default="/home/adarsh/software/meam517_final/data_v2/")

args = parser.parse_args()

net = Net(x_dim=0,
          u_dim=3,
          fcn_size_1=250,
          fcn_size_2=120,
          fcn_size_3=50).cuda().float()

net.load_state_dict(torch.load(args.net_path))

sample_fname = "/home/adarsh/software/meam517_final/data_v2/"
dset = TrajDataset(sample_fname, with_x=False, max_u=np.array([25, 25, 10]))

criterion = nn.L1Loss()

test_loader = DataLoader(dset, batch_size=1, num_workers=1, shuffle=True)

num_tests = 10

for i_batch, sample_batched in enumerate(test_loader):
    if i_batch == num_tests:
        break
    input, u1_out, u2_out, u3_out = sample_batched

    input = input.float().cuda()

    u1_out = u1_out.float()
    u2_out = u2_out.float()
    u3_out = u3_out.float()

    # forward!
    u1_pred, u2_pred, u3_pred = net(input)

    loss = criterion(u1_pred.cpu().float(), u1_out) + \
           criterion(u2_pred.cpu().float(), u2_out) + \
           criterion(u3_pred.cpu().float(), u3_out)

    print({'iteration': i_batch, 'loss': loss.item()})

    gen_plots(input, [u1_pred, u2_pred, u3_pred], [u1_out, u2_out, u3_out])
