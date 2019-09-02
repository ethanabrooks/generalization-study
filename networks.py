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
        x = F.relu(self.conv1(x))
        x1 = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x1))
        x = F.max_pool2d(x, 2, 2)
        x2 = x.view(-1, 4 * 4 * 50)
        x3 = F.relu(self.fc1(x2))
        x4 = self.fc2(x3)
        return F.log_softmax(x4, dim=1), x1, x2, x3, x4


class Discriminator(nn.Module):
    def __init__(self, hidden_size, num_hidden, activation, dropout):
        super().__init__()
        self.activation = (
            nn.Sequential(activation, nn.Dropout()) if dropout else activation
        )
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

    def forward(self, x1, x2, x3, x4):
        y1 = self.activation(self.conv1(x1))
        y1 = F.max_pool2d(y1, 2, 2)
        y1 = y1.view(-1, 4 * 4 * 50)
        y2 = self.activation(self.fc1(x2))
        y3 = self.activation(self.fc2(x3))
        y4 = self.activation(self.fc3(x4))
        return self.mlp(torch.cat([y1, y2, y3, y4], dim=-1))
