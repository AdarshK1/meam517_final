import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    def __init__(self, nought=35, x_dim=6, u_dim=3, fcn_size_1=300, fcn_size_2=400, with_pose=False):
        super(Net, self).__init__()

        self.with_pose = with_pose

        self.conv1_rgb_branch = nn.Conv2d(1, 5, kernel_size=3)
        self.conv2_rgb_branch = nn.Conv2d(5, 10, kernel_size=3)

        final_output_dim = nought * (x_dim + u_dim)

        self.fcn_1 = nn.Linear(10 * 11 * 11, fcn_size_1)
        self.fcn_2 = nn.Linear(fcn_size_1, fcn_size_2)
        self.fcn_3 = nn.Linear(fcn_size_2, final_output_dim)

    def forward(self, map, start=None, apex=None, goal=None):
        map = F.relu((self.conv1_rgb_branch(map)))
        map = F.relu((self.conv2_rgb_branch(map)))

        print(map.shape)
        out = torch.flatten(map, start_dim=1)
        print("flattened", out.shape)

        if self.with_pose:
            out = torch.cat([map, start, apex, goal], dim=1)
            print(out.shape)

        out = F.relu(self.fcn_1(out))
        out = F.relu(self.fcn_2(out))
        out = self.fcn_3(out)
        return out
