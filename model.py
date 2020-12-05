import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    def __init__(self, nought=35, x_dim=6, u_dim=3, fcn_size_1=300, fcn_size_2=400, fcn_size_3=400, with_pose=False):
        super(Net, self).__init__()

        self.with_pose = with_pose

        self.conv1 = nn.Conv2d(1, 3, kernel_size=3)
        self.conv2 = nn.Conv2d(3, 6, kernel_size=3)

        final_output_dim = nought * (x_dim + u_dim)

        self.fcn_1 = nn.Linear(6 * 11 * 11, fcn_size_1)
        self.fcn_2 = nn.Linear(fcn_size_1, fcn_size_2)
        self.fcn_3 = nn.Linear(fcn_size_2, fcn_size_3)
        self.fcn_4 = nn.Linear(fcn_size_3, final_output_dim)

    def forward(self, map, start=None, apex=None, goal=None):
        map = F.relu((self.conv1(map)))
        map = F.relu((self.conv2(map)))

        # print(map.shape)
        out = torch.flatten(map, start_dim=1)
        # print("flattened", out.shape)

        if self.with_pose:
            out = torch.cat([map, start, apex, goal], dim=1)
            # print(out.shape)

        out = F.relu(self.fcn_1(out))
        out = F.relu(self.fcn_2(out))
        out = F.relu(self.fcn_3(out))
        out = self.fcn_4(out)
        return out
