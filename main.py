from __future__ import print_function
import time
import itertools
from pathlib import Path
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from rl_utils import hierarchical_parse_args
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import random

from datasets import NoiseDataset, AddLabel
from networks import Classifier, Discriminator
from util import get_n_gpu, is_correct


def train(classifier, device, train_loader, optimizer, epoch, log_interval, writer):
    classifier.train()
    correct = 0
    total = 0
    start = time.time()
    for batch_idx, (data, (target, _)) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # TODO: add noise
        optimizer.zero_grad()
        output, *_ = classifier(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        correct += is_correct(output, target)
        total += target.numel()
        if batch_idx % log_interval == 0:
            idx = epoch * len(train_loader) + batch_idx

            tick = time.time()
            writer.add_scalar("fps", (tick - start) / log_interval, idx)
            start = tick

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


def test(classifier, device, test_loader, epoch, writer):
    classifier.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, (target, _) in test_loader:
            data, target = data.to(device), target.to(device)
            output, *_ = classifier(data)
            test_loss += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            correct += is_correct(output, target)

    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    writer.add_scalar("test accuracy", accuracy, epoch)
    writer.add_scalar("test loss", test_loss, epoch)
    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * accuracy
        )
    )


def train_discriminator(
    classifier,
    discriminator,
    device,
    train_loader,
    optimizer,
    epoch,
    log_interval,
    writer,
):
    classifier.eval()
    correct = 0
    total = 0
    for batch_idx, (data, (_, discriminator_target)) in enumerate(train_loader):
        data = data.to(device)
        discriminator_target = discriminator_target.to(device).unsqueeze(1).float()
        # TODO: add noise
        optimizer.zero_grad()
        classifier_output, *activations = classifier(data)
        discriminator_output = discriminator(*activations)
        loss = F.binary_cross_entropy_with_logits(
            discriminator_output, discriminator_target
        )
        loss.backward()
        optimizer.step()
        correct += (
            (torch.abs(discriminator_output.sigmoid() - discriminator_target) < 1e-4)
            .sum()
            .item()
        )
        total += discriminator_target.numel()
        if batch_idx % log_interval == 0:
            idx = epoch * len(train_loader) + batch_idx
            writer.add_scalar("disciminator train loss", loss.item(), idx)
            writer.add_scalar("discriminator train accuracy", correct / total, idx)
            print(
                "Discriminator Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )


def test_discriminator(classifier, discriminator, device, test_loader, epoch, writer):
    classifier.eval()
    test_loss = 0
    correct = 0
    total_error = 0
    with torch.no_grad():
        for data, (_, discriminator_target) in test_loader:
            data = data.to(device)
            discriminator_target = discriminator_target.to(device).unsqueeze(1).float()
            classifier_output, *activations = classifier(data)
            discriminator_output = discriminator(*activations)
            test_loss += F.binary_cross_entropy_with_logits(
                discriminator_output, discriminator_target
            )
            error = torch.abs(discriminator_output.sigmoid() - discriminator_target)
            total_error += error.mean()
            correct += (error < 1e-4).sum().item()

    N = len(test_loader.dataset)
    writer.add_scalar("discriminator test loss", test_loss / N, epoch)
    writer.add_scalar("discriminator test error", total_error / N, epoch)
    writer.add_scalar("discriminator test accuracy", correct / N, epoch)
    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss / N, correct, N, 100.0 * correct / N
        )
    )


def main(
    no_cuda,
    seed,
    batch_size,
    percent_noise,
    test_batch_size,
    optimizer_args,
    classifier_epochs,
    discriminator_epochs,
    discriminator_args,
    classifier_load_path,
    log_dir,
    log_interval,
    run_id,
):
    use_cuda = not no_cuda and torch.cuda.is_available()
    torch.manual_seed(1)
    if use_cuda:
        n_gpu = get_n_gpu()
        try:
            index = int(run_id[-1])
        except ValueError:
            index = random.randrange(0, n_gpu)
        device = torch.device("cuda", index=index % n_gpu)
    else:
        device = "cpu"
    kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True} if use_cuda else {}

    train_dataset = NoiseDataset(
        "../data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
        percent_noise=percent_noise,
    )
    test_dataset = NoiseDataset(
        "../data",
        train=False,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
        percent_noise=percent_noise,
    )
    half = (len(train_dataset) + len(test_dataset)) // 2
    train_dataset, test_dataset = random_split(
        train_dataset + test_dataset, [half, half]
    )
    train_dataset = AddLabel(train_dataset, 0)
    test_dataset = AddLabel(test_dataset, 0)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, **kwargs)
    classifier = Classifier().to(device)
    optimizer = optim.SGD(classifier.parameters(), **optimizer_args)
    writer = SummaryWriter(str(log_dir))
    if classifier_load_path:
        classifier.load_state_dict(torch.load(classifier_load_path))
        for epoch in range(1, classifier_epochs + 1):
            test(
                classifier=classifier,
                device=device,
                test_loader=train_loader,
                epoch=epoch,
                writer=writer,
            )
        torch.manual_seed(seed)
    else:
        torch.manual_seed(seed)
        for epoch in range(1, classifier_epochs + 1):
            train(
                classifier=classifier,
                device=device,
                train_loader=train_loader,
                optimizer=optimizer,
                epoch=epoch,
                log_interval=log_interval,
                writer=writer,
            )
            test(
                classifier=classifier,
                device=device,
                test_loader=test_loader,
                epoch=epoch,
                writer=writer,
            )
        torch.save(classifier.state_dict(), str(Path(log_dir, "mnist_cnn.pt")))

    discriminator = Discriminator(**discriminator_args).to(device)
    train_dataset, test_dataset = random_split(
        train_dataset + test_dataset, [half, half]
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, **kwargs)
    optimizer = optim.SGD(discriminator.parameters(), **optimizer_args)
    iterator = (
        range(1, discriminator_epochs + 1)
        if discriminator_epochs
        else itertools.count()
    )
    for epoch in iterator:
        train_discriminator(
            classifier=classifier,
            discriminator=discriminator,
            device=device,
            train_loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            log_interval=log_interval,
            writer=writer,
        )
        test_discriminator(
            classifier=classifier,
            discriminator=discriminator,
            device=device,
            test_loader=test_loader,
            epoch=epoch,
            writer=writer,
        )


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
    parser.add_argument("--percent-noise", type=float, required=True, metavar="N")
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--classifier-epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--discriminator-epochs",
        type=int,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    discriminator_parser = parser.add_argument_group("discriminator_args")
    discriminator_parser.add_argument(
        "--hidden-size", type=int, default=512, metavar="N"
    )
    discriminator_parser.add_argument("--num-hidden", type=int, default=1, metavar="N")
    discriminator_parser.add_argument(
        "--activation", type=lambda s: eval(f"nn.{s}"), default=nn.ReLU(), metavar="N"
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
    parser.add_argument("--run-id", metavar="N", default="")
    parser.add_argument("--classifier-load-path")
    main(**hierarchical_parse_args(parser))


if __name__ == "__main__":
    cli()
