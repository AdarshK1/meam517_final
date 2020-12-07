import torch
import torch.nn as nn
import numpy as np
from model import Net
from dataloader import TrajDataset
from torch.utils.data import DataLoader
import time
import argparse
import matplotlib.pyplot as plt


def gen_plots(input, gt, pred, u_max=np.array([25, 25, 10])):
    input = input[0, 0, :, :].cpu()
    gt = gt[0, :]
    pred = pred[0, :]

    print(input.shape, gt.shape, pred.shape)
    u1_gt = []
    u1_pred = []
    u2_gt = []
    u2_pred = []
    u3_gt = []
    u3_pred = []

    for i in range(gt.shape[0]):
        if i % 3 == 0:
            u1_gt.append(gt[i] * u_max[0])
            u1_pred.append(pred[i] * u_max[0])
        elif i % 3 == 1:
            u2_gt.append(gt[i] * u_max[1])
            u2_pred.append(pred[i] * u_max[1])
        elif i % 3 == 2:
            u3_gt.append(gt[i] * u_max[2])
            u3_pred.append(pred[i] * u_max[2])

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
          fcn_size_1=1000,
          fcn_size_2=1000,
          fcn_size_3=500).cuda().float()

net.load_state_dict(torch.load(args.net_path))

sample_fname = "/home/adarsh/software/meam517_final/data_v2/"
dset = TrajDataset(sample_fname, with_x=False, max_u=75)

criterion = nn.L1Loss()

test_loader = DataLoader(dset, batch_size=1, num_workers=1, shuffle=True)

num_tests = 1

for i_batch, sample_batched in enumerate(test_loader):
    if i_batch == num_tests:
        break
    input, output = sample_batched
    input = input.float().cuda()
    output = output.float()

    output_predicted = net(input)

    loss = criterion(output_predicted.cpu().float(), output)

    print({'iteration': i_batch, 'loss': loss.item()})

    gen_plots(input, output, output_predicted)
