import torch

from common import Config
from model_factory.Mobile.models import load_mobile_net_v2
from model_factory.hub_common_profile import common_train_template, common_inference_template, common_checkpoint_template
from profiler_utils import Profileable, ProfileIterator


def mobile_rand_input(batch_size: int):
    return torch.rand((batch_size, 3, 224, 224), device=Config().device)


def mobile_rand_output():
    return torch.rand((1,), device=Config().device)


class MobileNetV2Train(Profileable):
    def profile(self, batch_size: int, duration_sec: int) -> ProfileIterator:
        model = load_mobile_net_v2()
        return common_train_template(model, batch_size, duration_sec, mobile_rand_input, mobile_rand_output)


class MobileNetV2Inference(Profileable):
    def profile(self, batch_size: int, duration_sec: int) -> ProfileIterator:
        model = load_mobile_net_v2()
        return common_inference_template(model, batch_size, duration_sec, mobile_rand_input)


class MobileNetV2Checkpoint(Profileable):
    def profile(self, batch_size: int, duration_sec: int) -> ProfileIterator:
        model = load_mobile_net_v2()
        common_train_template(model, batch_size, 5, mobile_rand_input, mobile_rand_output)
        return common_checkpoint_template(model, duration_sec)

def do_test():
    from common import process_group
    process_group.setup(0, 1)
    profiler = MobileNetV2Train()
    iterator = profiler.profile(batch_size=16, duration_sec=10)
    print(iterator.to_dict())
    torch.cuda.empty_cache()
    profiler = MobileNetV2Inference()
    iterator = profiler.profile(batch_size=16, duration_sec=10)
    print(iterator.to_dict())


if __name__ == '__main__':
    do_test()
