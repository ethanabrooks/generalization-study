import torch
from torch.utils.data import Dataset, ConcatDataset
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
    def __init__(self, dataset, extra_label, random_labels=False):
        self.label = extra_label
        self.dataset = dataset
        self.random_labels = (
            torch.randint(low=0, high=2, size=(len(dataset),))
            if random_labels
            else None
        )

    def __getitem__(self, item):
        x, y = self.dataset[item]
        try:
            label = self.random_labels[item]
        except TypeError:
            label = self.label
        return x, (y, label)

    def __len__(self):
        return len(self.dataset)


class SimpleDataset(Dataset):
    def __init__(self, n: int, generalization_error: float):
        self.targets = torch.randint(low=0, high=2, size=(n,))
        generalization_errors = torch.bernoulli(
            generalization_error * torch.ones(n).float()
        )
        self.generalization_bits = torch.abs(
            self.targets.float() - generalization_errors
        ).unsqueeze(1)
        self.one_hots = torch.eye(n).float()

    def __getitem__(self, item):
        target = self.targets[item]
        generalization_bit = self.generalization_bits[item]
        unique_code = self.one_hots[item]
        return torch.cat([generalization_bit, unique_code], dim=-1), target

    def __len__(self):
        return self.targets.numel()
