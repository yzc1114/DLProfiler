import pathlib

import torch

from .path import nvidia_path

def load_efficient_net():
    return torch.hub.load(nvidia_path, "nvidia_efficientnet_b0", pretrained=False, source="local")


def do_test():
    model1 = load_efficient_net()
    print(model1)


if __name__ == '__main__':
    do_test()
