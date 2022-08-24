import itertools
from typing import Callable

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from common import Config
from profiler_utils import ProfileIterator


def common_train_template(model: torch.nn.Module, batch_size: int, duration_sec: int,
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
        pred = torch.mean(torch.flatten(pred))
        loss = pred - y
        loss.backward()
        optimizer.step()
    return iterator


def common_inference_template(model: torch.nn.Module, batch_size: int, duration_sec: int,
                              rand_input: Callable[[int], torch.Tensor]):
    model.to(Config().local_rank)
    model.train(False)
    with torch.no_grad():
        iterator = ProfileIterator(itertools.count(0), duration_sec)
        for _ in iterator:
            X = rand_input(batch_size)
            model(X)
    return iterator
