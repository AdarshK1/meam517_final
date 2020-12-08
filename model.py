import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    def __init__(self, nought=35, x_dim=6, u_dim=3, fcn_size_1=300, fcn_size_2=400, fcn_size_3=400, fcn_size_4=400,
                 with_pose=False):
        super(Net, self).__init__()

        self.with_pose = with_pose

        self.conv1 = nn.Conv2d(1, 3, kernel_size=5)
        self.conv2 = nn.Conv2d(3, 6, kernel_size=3)

        final_output_dim = nought * (x_dim + u_dim)

        self.fcn_1_u1 = nn.Linear(6 * 9 * 9, fcn_size_1)
        self.fcn_2_u1 = nn.Linear(fcn_size_1, fcn_size_2)
        self.fcn_3_u1 = nn.Linear(fcn_size_2, fcn_size_3)
        self.fcn_4_u1 = nn.Linear(fcn_size_3 * 3, fcn_size_4)
        self.fcn_5_u1 = nn.Linear(fcn_size_4, nought)

        self.fcn_1_u2 = nn.Linear(6 * 9 * 9, fcn_size_1)
        self.fcn_2_u2 = nn.Linear(fcn_size_1, fcn_size_2)
        self.fcn_3_u2 = nn.Linear(fcn_size_2, fcn_size_3)
        self.fcn_4_u2 = nn.Linear(fcn_size_3 * 3, fcn_size_4)
        self.fcn_5_u2 = nn.Linear(fcn_size_4, nought)

        self.fcn_1_u3 = nn.Linear(6 * 9 * 9, fcn_size_1)
        self.fcn_2_u3 = nn.Linear(fcn_size_1, fcn_size_2)
        self.fcn_3_u3 = nn.Linear(fcn_size_2, fcn_size_3)
        self.fcn_4_u3 = nn.Linear(fcn_size_3 * 3, fcn_size_4)
        self.fcn_5_u3 = nn.Linear(fcn_size_4, nought)

    def forward(self, map, start=None, apex=None, goal=None):
        map = F.relu((self.conv1(map)))
        map = F.relu((self.conv2(map)))

        # print(map.shape)
        out = torch.flatten(map, start_dim=1)
        # print("flattened", out.shape)

        if self.with_pose:
            out = torch.cat([map, start, apex, goal], dim=1)
            # print(out.shape)

        u1_out = F.relu(self.fcn_1_u1(out))
        u1_out = F.relu(self.fcn_2_u1(u1_out))
        u1_out = F.relu(self.fcn_3_u1(u1_out))

        u2_out = F.relu(self.fcn_1_u2(out))
        u2_out = F.relu(self.fcn_2_u2(u2_out))
        u2_out = F.relu(self.fcn_3_u2(u2_out))

        u3_out = F.relu(self.fcn_1_u3(out))
        u3_out = F.relu(self.fcn_2_u3(u3_out))
        u3_out = F.relu(self.fcn_3_u3(u3_out))

        # print(u3_out.shape)
        concated = torch.cat([u1_out, u2_out, u3_out], dim=1)
        # print(concated.shape)
        u1_out = F.tanh(self.fcn_4_u1(concated))
        u1_out = self.fcn_5_u1(u1_out)
        u2_out = F.tanh(self.fcn_4_u2(concated))
        u2_out = self.fcn_5_u2(u2_out)
        u3_out = F.tanh(self.fcn_4_u3(concated))
        u3_out = self.fcn_5_u3(u3_out)

        return u1_out, u2_out, u3_out


class FeasibilityNet(nn.Module):
    def __init__(self, fcn_size_1=300, fcn_size_2=400, fcn_size_3=400, fcn_size_4=400, fcn_size_5=400):
        super(FeasibilityNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 3, kernel_size=5)
        self.conv2 = nn.Conv2d(3, 6, kernel_size=3)

        self.fcn_1 = nn.Linear(6 * 9 * 9, fcn_size_1)
        self.fcn_2 = nn.Linear(fcn_size_1, fcn_size_2)
        self.fcn_3 = nn.Linear(fcn_size_2, fcn_size_3)
        self.fcn_4 = nn.Linear(fcn_size_3, fcn_size_4)
        self.fcn_5 = nn.Linear(fcn_size_4, fcn_size_5)
        self.fcn_6 = nn.Linear(fcn_size_5, 1)

    def forward(self, map):
        map = F.relu((self.conv1(map)))
        map = F.relu((self.conv2(map)))

        out = torch.flatten(map, start_dim=1)

        out = F.relu(self.fcn_1(out))
        out = F.relu(self.fcn_2(out))
        out = F.relu(self.fcn_3(out))
        out = F.relu(self.fcn_4(out))
        out = F.relu(self.fcn_5(out))
        out = F.sigmoid(self.fcn_6(out))

        return out
