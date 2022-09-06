import pathlib

import torch

pytorch_vision_path = str(pathlib.Path(__file__).parent.parent / "repos" / "pytorch_vision_v0.10.0")


def load_vgg16():
    return torch.hub.load(pytorch_vision_path, "vgg16", pretrained=False, source="local")

def load_vgg19():
    return torch.hub.load(pytorch_vision_path, "vgg19", pretrained=False, source="local")


def do_test():
    model1 = load_vgg16()
    print(model1)
    model1 = load_vgg19()
    print(model1)


if __name__ == '__main__':
    do_test()
