import pathlib

import torch

pytorch_vision_path = str(pathlib.Path(__file__).parent.parent / "repos" / "pytorch_vision_v0.10.0")


def load_mobile_net_v2():
    return torch.hub.load(pytorch_vision_path, "mobilenet_v2", pretrained=False, source="local")


def do_test():
    model1 = load_mobile_net_v2()
    print(model1)


if __name__ == '__main__':
    do_test()
