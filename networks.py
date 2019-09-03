import torch
from torch import nn as nn
from torch.nn import functional as F


class Classifier(nn.Module):
    def __init__(self, n: int):
        super(Classifier, self).__init__()
        self.simple = bool(n)
        if self.simple:
            self.fc = nn.Linear(n + 1, 1)
        else:
            self.conv1 = nn.Conv2d(1, 20, 5, 1)
            self.conv2 = nn.Conv2d(20, 50, 5, 1)
            self.fc1 = nn.Linear(4 * 4 * 50, 500)
            self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        if self.simple:
            out = F.sigmoid(self.fc(x))
            return torch.cat([1 - out, out], dim=-1), self.fc.weight * x
        else:
            x = F.relu(self.conv1(x))
            x1 = F.max_pool2d(x, 2, 2)
            x = F.relu(self.conv2(x1))
            x = F.max_pool2d(x, 2, 2)
            x2 = x.view(-1, 4 * 4 * 50)
            x3 = F.relu(self.fc1(x2))
            x4 = self.fc2(x3)
            return F.log_softmax(x4, dim=1), x1, x2, x3, x4


class Discriminator(nn.Module):
    def __init__(self, hidden_size, num_hidden, activation, n):
        super().__init__()
        self.simple = bool(n)
        if self.simple:
            self.fc = nn.Linear(n + 1, 1)
        else:
            self.activation = activation
            self.conv1 = nn.Conv2d(20, 50, 5, 1)
            self.fc1 = nn.Linear(4 * 4 * 50, hidden_size)
            self.fc2 = nn.Linear(500, hidden_size)
            self.fc3 = nn.Linear(10, hidden_size)
            in_size = 4 * 4 * 50 + hidden_size * 3
            layers = []
            for _ in range(num_hidden):
                layers += [nn.Linear(in_size, hidden_size), activation]
                in_size = hidden_size
            layers += [nn.Linear(in_size, 1)]
            self.mlp = nn.Sequential(*layers)

    def forward(self, *xs):
        if self.simple:
            return self.fc(*xs)
        else:
            x1, x2, x3, x4 = xs
            y1 = self.activation(self.conv1(x1))
            y1 = F.max_pool2d(y1, 2, 2)
            y1 = y1.view(-1, 4 * 4 * 50)
            y2 = self.activation(self.fc1(x2))
            y3 = self.activation(self.fc2(x3))
            y4 = self.activation(self.fc3(x4))
            return self.mlp(torch.cat([y1, y2, y3, y4], dim=-1))
