import pathlib

import torch

pytorch_vision_path = str(pathlib.Path(__file__).parent.parent / "repos" / "pytorch_vision_v0.10.0")


def load_resnet50():
    return torch.hub.load(pytorch_vision_path, "resnet50", pretrained=False, source="local")


def load_resnet18():
    return torch.hub.load(pytorch_vision_path, "resnet18", pretrained=False, source="local")


def do_test():
    model1 = load_resnet18()
    model2 = load_resnet50()
    print(model1, model2)


if __name__ == '__main__':
    do_test()
