import torch
from torch import nn as nn
from torch.nn import functional as F


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        N = x.size(0)
        x1 = F.relu(self.conv1(x))
        x2 = F.max_pool2d(x1, 2, 2)
        x3 = F.relu(self.conv2(x2))
        x4 = F.max_pool2d(x3, 2, 2)
        x5 = x4.view(-1, 4 * 4 * 50)
        x6 = F.relu(self.fc1(x5))
        x7 = self.fc2(x6)
        activations = torch.cat(
            [x.view(N, -1) for x in [x1, x2, x3, x4, x5, x6, x7]], dim=-1
        )
        return F.log_softmax(x7, dim=1), activations


class Discriminator(nn.Module):
    def __init__(self, hidden_sizes, activation):
        super().__init__()
        sizes = [19710] + list(hidden_sizes)
        modules = []
        self.net = nn.Sequential(
            *[
                nn.Sequential(nn.Linear(in_size, out_size), activation)
                for in_size, out_size in zip(sizes, sizes[1:])
            ],
            nn.Linear(sizes[-1], 1),
        )

    def forward(self, x):
        return self.net(x)
