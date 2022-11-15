import itertools
from typing import Callable

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from common import Config
from model_factory.Yolo.models import load_yolov5s
from model_factory.hub_common_profile import common_inference_template, common_checkpoint_template
from profiler_utils import Profileable, ProfileIterator


def yolov5s_rand_input(batch_size: int):
    return torch.rand([batch_size, 3, 256, 256], device=Config().device)


def yolov5s_rand_output():
    return torch.rand((1,), device=Config().device)


def yolov5_train_template(model: torch.nn.Module, batch_size: int, duration_sec: int,
                          rand_input: Callable[[int], torch.Tensor], rand_output: Callable[[], torch.Tensor]):
    model = model.to(Config().device)
    model = DDP(model)
    model.train()
    iterator = ProfileIterator(itertools.count(0), duration_sec)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in iterator:
        optimizer.zero_grad()
        (X, y) = (rand_input(batch_size), rand_output())
        pred = model(X)
        mean = None
        for each in pred:
            each_mean = torch.mean(torch.flatten(each))
            if mean is None:
                mean = each_mean
            else:
                mean += each_mean
        pred = mean
        loss = pred - y
        loss.backward()
        optimizer.step()
    return iterator


class YoloV5sTrain(Profileable):
    def profile(self, batch_size: int, duration_sec: int) -> ProfileIterator:
        model = load_yolov5s()
        return yolov5_train_template(model, batch_size, duration_sec, yolov5s_rand_input, yolov5s_rand_output)


class YoloV5sInference(Profileable):
    def profile(self, batch_size: int, duration_sec: int) -> ProfileIterator:
        model = load_yolov5s()
        return common_inference_template(model, batch_size, duration_sec, yolov5s_rand_input)



class YoloV5sCheckpoint(Profileable):
    def profile(self, batch_size: int, duration_sec: int) -> ProfileIterator:
        model = load_yolov5s()
        yolov5_train_template(model, batch_size, duration_sec, yolov5s_rand_input, yolov5s_rand_output)
        return common_checkpoint_template(model, duration_sec)


def do_test():
    from common import process_group
    # process_group.setup(0, 1)
    profile = YoloV5sCheckpoint()
    iterator = profile.profile(batch_size=64, duration_sec=10)
    print(iterator.to_dict())
    # torch.cuda.empty_cache()
    # profile = YoloV5sInference()
    # iterator = profile.profile(batch_size=64, duration_sec=10)
    # print(iterator.to_dict())


if __name__ == '__main__':
    do_test()
