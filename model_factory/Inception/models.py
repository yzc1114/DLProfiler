import pathlib

import torch

pytorch_vision_path = str(pathlib.Path(__file__).parent.parent / "repos" / "pytorch_vision_v0.10.0")


def load_inception_v3():
    return torch.hub.load(pytorch_vision_path, "inception_v3", pretrained=False, source="local")


def do_test():
    model1 = load_inception_v3()
    print(model1)


if __name__ == '__main__':
    do_test()
