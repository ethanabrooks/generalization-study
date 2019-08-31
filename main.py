from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from rl_utils import hierarchical_parse_args
from tensorboardX import SummaryWriter
from torchvision import datasets, transforms


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(model, device, train_loader, optimizer, epoch, log_interval, writer):
    model.train()
    correct = 0
    total = 0
    for batch_idx, (data, (target, _)) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # TODO: add noise
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        correct += is_correct(output, target)
        total += target.numel()
        if batch_idx % log_interval == 0:
            idx = epoch * len(train_loader) + batch_idx
            writer.add_scalar("loss", loss.item(), idx)
            writer.add_scalar("train accuracy", correct / total, idx)
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )


def test(model, device, test_loader, epoch, writer):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, (target, _) in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            correct += is_correct(output, target)

    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    writer.add_scalar("accuracy", accuracy, epoch)
    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * accuracy
        )
    )


def is_correct(output, target):
    pred = output.argmax(
        dim=1, keepdim=True
    )  # get the index of the max log-probability
    return pred.eq(target.view_as(pred)).sum().item()


def main(
    no_cuda,
    seed,
    batch_size,
    test_batch_size,
    optimizer_args,
    epochs,
    save_model,
    log_dir,
    log_interval,
):
    use_cuda = not no_cuda and torch.cuda.is_available()
    torch.manual_seed(seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    # TODO: use target transform to label train vs test.
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../data",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
            target_transform=lambda t: (t, 0),
        ),
        batch_size=batch_size,
        shuffle=True,
        **kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../data",
            train=False,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
            target_transform=lambda t: (t, 1),
        ),
        batch_size=test_batch_size,
        shuffle=True,
        **kwargs
    )
    # TODO create mixed dataset with train/test labels
    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), **optimizer_args)
    writer = SummaryWriter(str(log_dir))
    for epoch in range(1, epochs + 1):
        train(
            model=model,
            device=device,
            train_loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            log_interval=log_interval,
            writer=writer,
        )
        test(
            model=model,
            device=device,
            test_loader=test_loader,
            epoch=epoch,
            writer=writer,
        )
    # TODO: train discriminator
    if save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


def cli():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    optimizer_parser = parser.add_argument_group("optimizer_args")
    optimizer_parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        metavar="LR",
        help="learning rate (default: 0.01)",
    )
    optimizer_parser.add_argument(
        "--momentum",
        type=float,
        default=0.5,
        metavar="M",
        help="SGD momentum (default: 0.5)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument("--log-dir", default="/tmp/mnist", metavar="N")

    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )
    main(**hierarchical_parse_args(parser))


if __name__ == "__main__":
    cli()
