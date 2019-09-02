import torch
from torch.utils.data import Dataset
from torchvision import datasets


class NoiseDataset(datasets.MNIST):
    def __init__(self, *args, percent_noise=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.percent_noise = percent_noise
        self.noise = torch.randn(self.data.shape)

    def __getitem__(self, index):
        x, y = super().__getitem__(index)
        noise = self.noise[index]
        x = noise * self.percent_noise + x * (1 - self.percent_noise)
        return x, y


class AddLabel(Dataset):
    def __init__(self, dataset, extra_label):
        self.label = extra_label
        self.dataset = dataset

    def __getitem__(self, item):
        x, y = self.dataset[item]
        return x, (y, self.label)

    def __add__(self, other):
        return self.dataset + other

    def __len__(self):
        return len(self.dataset)
