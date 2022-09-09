from model_factory.Inception.models import load_inception_v3
from model_factory.hub_common_profile import common_train_template

import itertools
from typing import Callable

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from common import Config
from model_factory.hub_common_profile import common_inference_template
from profiler_utils import Profileable, ProfileIterator


def inception_v3_rand_input(batch_size: int):
    return torch.rand((batch_size, 3, 299, 299), device=Config().local_rank)


def inception_v3_rand_output():
    return torch.rand((1,), device=Config().local_rank)


def inception_train_template(model: torch.nn.Module, batch_size: int, duration_sec: int,
                          rand_input: Callable[[int], torch.Tensor], rand_output: Callable[[], torch.Tensor]):
    model = model.to(Config().local_rank)
    model = DDP(model)
    model.train()
    iterator = ProfileIterator(itertools.count(0), duration_sec)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in iterator:
        optimizer.zero_grad()
        (X, y) = (rand_input(batch_size), rand_output())
        pred = model(X)
        pred = torch.mean(torch.flatten(pred[0]) + torch.flatten(pred[1]))
        loss = pred - y
        loss.backward()
        optimizer.step()
    return iterator


class InceptionV3Train(Profileable):
    def profile(self, batch_size: int, duration_sec: int) -> ProfileIterator:
        model = load_inception_v3()
        return inception_train_template(model, batch_size, duration_sec, inception_v3_rand_input, inception_v3_rand_output)


class InceptionV3Inference(Profileable):
    def profile(self, batch_size: int, duration_sec: int) -> ProfileIterator:
        model = load_inception_v3()
        return common_inference_template(model, batch_size, duration_sec, inception_v3_rand_input)


def do_test():
    from common import process_group
    process_group.setup(0, 1)
    profiler = InceptionV3Train()
    iterator = profiler.profile(batch_size=16, duration_sec=10)
    print(iterator.to_dict())
    torch.cuda.empty_cache()
    profiler = InceptionV3Inference()
    iterator = profiler.profile(batch_size=16, duration_sec=10)
    print(iterator.to_dict())


if __name__ == '__main__':
    do_test()
