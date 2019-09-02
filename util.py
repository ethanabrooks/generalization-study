import csv
import subprocess
from io import StringIO

from numpy import argmax


def get_n_gpu():
    nvidia_smi = subprocess.check_output(
        "nvidia-smi --format=csv --query-gpu=memory.free".split(),
        universal_newlines=True,
    )
    return len(list(csv.reader(StringIO(nvidia_smi)))) - 1


def get_freer_gpu():
    nvidia_smi = subprocess.check_output(
        "nvidia-smi --format=csv --query-gpu=memory.free".split(),
        universal_newlines=True,
    )
    free_memory = [
        float(x[0].split()[0])
        for i, x in enumerate(csv.reader(StringIO(nvidia_smi)))
        if i > 0
    ]
    return int(argmax(free_memory))


def is_correct(output, target):
    pred = output.argmax(
        dim=1, keepdim=True
    )  # get the index of the max log-probability
    return pred.eq(target.view_as(pred)).sum().item()


def binary_is_correct(output, target):
    return output.sigmoid().round().eq(target).sum().item()
