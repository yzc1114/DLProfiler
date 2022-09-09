import pathlib

from model_factory.EfficientNet.models import load_efficient_net
from model_factory.hub_common_profile import common_train_template
import torch

from common import Config
from model_factory.hub_common_profile import common_inference_template
from profiler_utils import Profileable, ProfileIterator

from .path import nvidia_path

utils = torch.hub.load(nvidia_path, "nvidia_convnets_processing_utils", source="local")

image_input = utils.prepare_input_from_uri(str(pathlib.Path(__file__).parent / "test.jpg"))

def efficient_net_rand_input(batch_size: int):
    return torch.cat([image_input for _ in range(batch_size)]).to(Config().local_rank)


def efficient_net_rand_output():
    return torch.rand((1,), device=Config().local_rank)


class EfficientNetTrain(Profileable):
    def profile(self, batch_size: int, duration_sec: int) -> ProfileIterator:
        model = load_efficient_net()
        return common_train_template(model, batch_size, duration_sec, efficient_net_rand_input, efficient_net_rand_output)


class EfficientNetInference(Profileable):
    def profile(self, batch_size: int, duration_sec: int) -> ProfileIterator:
        model = load_efficient_net()
        return common_inference_template(model, batch_size, duration_sec, efficient_net_rand_input)


def do_test():
    from common import process_group
    process_group.setup(0, 1)
    profiler = EfficientNetTrain()
    iterator = profiler.profile(batch_size=16, duration_sec=10)
    print(iterator.to_dict())
    torch.cuda.empty_cache()
    profiler = EfficientNetInference()
    iterator = profiler.profile(batch_size=16, duration_sec=10)
    print(iterator.to_dict())


if __name__ == '__main__':
    do_test()
