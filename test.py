import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import wandb
from model import Net, FeasibilityNet, CoefNet
from dataloader import TrajDataset
from torch.utils.data import DataLoader
import time
import sys
from datetime import datetime
import os

from pydrake.polynomial import Polynomial
from pydrake.trajectories import PiecewisePolynomial

import matplotlib.pyplot as plt

from tkinter import Tk
Tk().withdraw()
from tkinter.filedialog import askopenfilename


def test_u_coef(model_path):
    t0 = 0
    tf = 2
    N = 35

    n_test = 3

    net = CoefNet().cuda().float()
    net.load_state_dict(torch.load(model_path))
    net.eval()

    sample_fname = "/home/austin/repos/meam517_final/data_v3_test/"
    dset = TrajDataset(sample_fname, u_coef_classifier=True)

    for i in range(n_test):
        # plt.figure()
        input, u1_coef, u2_coef, u3_coef = dset.__getitem__(i)

        u1_coef = u1_coef.reshape(-1, 2)
        u2_coef = u2_coef.reshape(-1, 2)
        u3_coef = u3_coef.reshape(-1, 2)
        n_segments = u1_coef.shape[0]

        input = torch.Tensor(input).unsqueeze(0)

        input = input.float().cuda()

        with torch.no_grad():
            u1_pred, u2_pred, u3_pred = net(input)

        u1_pred = u1_pred.reshape(-1, 2).cpu().numpy()
        u2_pred = u2_pred.reshape(-1, 2).cpu().numpy()
        u3_pred = u3_pred.reshape(-1, 2).cpu().numpy()

        breaks = list(np.linspace(t0, tf, n_segments+1))

        u1_poly_list = []
        u2_poly_list = []
        u3_poly_list = []

        pred_u1_list = []
        pred_u2_list = []
        pred_u3_list = []

        for j in range(n_segments):
            u1_poly = Polynomial(u1_coef[j, :])
            u2_poly = Polynomial(u2_coef[j, :])
            u3_poly = Polynomial(u3_coef[j, :])

            u1_poly_list.append(u1_poly)
            u2_poly_list.append(u2_poly)
            u3_poly_list.append(u3_poly)

            pred_u1_poly = Polynomial(u1_pred[j, :])
            pred_u2_poly = Polynomial(u2_pred[j, :])
            pred_u3_poly = Polynomial(u3_pred[j, :])

            pred_u1_list.append(pred_u1_poly)
            pred_u2_list.append(pred_u2_poly)
            pred_u3_list.append(pred_u3_poly)


        # actual_traj = PiecewisePolynomial(actual_poly_list, breaks)
        u1_traj = PiecewisePolynomial(u1_poly_list, breaks)
        u2_traj = PiecewisePolynomial(u2_poly_list, breaks)
        u3_traj = PiecewisePolynomial(u3_poly_list, breaks)

        pred_u1_traj = PiecewisePolynomial(pred_u1_list, breaks)
        pred_u2_traj = PiecewisePolynomial(pred_u2_list, breaks)
        pred_u3_traj = PiecewisePolynomial(pred_u3_list, breaks)

        t = np.linspace(t0, tf, 100)

        u1_vals = [u1_traj.value(time).item() for time in t]
        pred_u1_vals = [pred_u1_traj.value(time).item() for time in t]

        u2_vals = [u2_traj.value(time).item() for time in t]
        pred_u2_vals = [pred_u2_traj.value(time).item() for time in t]

        u3_vals = [u3_traj.value(time).item() for time in t]
        pred_u3_vals = [pred_u3_traj.value(time).item() for time in t]


        fig, ax = plt.subplots(4, 1)
        fig.tight_layout(h_pad=2)

        plt.subplot(4,1,1)
        plt.title("u1")
        plt.plot(t, u1_vals, t, pred_u1_vals)
        plt.legend(["Actual", "Predicted"])

        plt.subplot(4,1,2)
        plt.title("u2")
        plt.plot(t, u2_vals, t, pred_u2_vals)
        plt.legend(["Actual", "Predicted"])

        plt.subplot(4,1,3)
        plt.title("u3")
        plt.plot(t, u3_vals, t, pred_u3_vals)
        plt.legend(["Actual", "Predicted"])

        input = input[0, 0, :, :].cpu()
        # print(input.shape)
        plt.subplot(4,1,4)
        plt.imshow(input)

        plt.savefig("u_prediction-"+str(i)+".png", dpi=600)

    plt.show()



if __name__=="__main__":
    # model_path = "/home/austin/repos/meam517_final/models/12_10_16_30_feasibility_classifier/model.ckpt"
    model_path = askopenfilename(title="Select model to test")

    test_u_coef(model_path)
