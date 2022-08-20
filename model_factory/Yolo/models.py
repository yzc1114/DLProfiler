import pathlib

import torch

# yolov5 dependencies check

yolov5_path = str(pathlib.Path(__file__).parent.parent / "repos" / "ultralytics_yolov5_master")


def load_yolov5s():
    return torch.hub.load(yolov5_path, "yolov5s", pretrained=False, source="local", autoshape=False)


def do_test():
    model1 = load_yolov5s()
    print(model1)


if __name__ == '__main__':
    do_test()
