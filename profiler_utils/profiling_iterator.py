import time
from common import time_ns
from abc import ABC, abstractmethod
from collections.abc import Iterable
from threading import Thread
from typing import Optional

import numpy as np
import torch.cuda

from common import Config


class ProfileIterator:
    def __init__(self, wrapped_iterator_object: Iterable, duration_sec: int):
        self.wrapped_iterator_object = wrapped_iterator_object
        self.duration_sec: int = duration_sec
        self.__init_profiling_variables()
        self.extra_dict = dict()

    def __init_profiling_variables(self):
        self.iteration_intervals = []
        self.mem_infos = []
        self.utilization = []
        self.last_iteration_time = 0
        self.wrapped_iterator = self.wrapped_iterator_object.__iter__()
        self.start_iteration_time = time_ns()
        self.monitor_thread: Optional[Thread] = None

    def __iter__(self):
        self.__init_profiling_variables()
        self.__spawn_mem_utilization_monitor_thread()
        return self

    def __spawn_mem_utilization_monitor_thread(self):
        t = Thread(target=self.mem_utilization_monitor)
        self.monitor_thread = t
        self.monitor_thread.start()

    def mem_utilization_monitor(self):
        while time_ns() - self.start_iteration_time < self.duration_sec * 1e9:
            self.mem_infos.append(list(torch.cuda.mem_get_info(Config().device)))
            self.utilization.append(torch.cuda.utilization(Config().device))
            time.sleep(Config().mem_utilization_monitor_interval)

    def __next__(self):
        now_ns = time_ns()
        if self.last_iteration_time != 0:
            self.iteration_intervals.append(now_ns - self.last_iteration_time)
        self.last_iteration_time = now_ns
        if now_ns - self.start_iteration_time > self.duration_sec * 1e9:
            self.monitor_thread.join()
            raise StopIteration
        return next(self.wrapped_iterator)

    def to_dict(self):
        return {
            "iteration_count": len(self.iteration_intervals),
            "iteration_intervals": self.iteration_intervals,
            "iteration_intervals_avg": np.mean(self.iteration_intervals) if len(self.iteration_intervals) > 0 else 0,
            "total_time_ns": self.last_iteration_time - self.start_iteration_time,
            "mem_infos": self.mem_infos,
            "utilization": self.utilization,
            "extra_dict": self.extra_dict
        }


class Profileable(ABC):
    @abstractmethod
    def profile(self, batch_size: int, duration_sec: int) -> ProfileIterator:
        pass


def do_test():
    ls = [1, 2, 3, 4]
    profile_iter = ProfileIterator(ls, 10)
    for i in profile_iter:
        time.sleep(1)
        print(i)
    print(profile_iter.to_dict())


if __name__ == '__main__':
    do_test()
