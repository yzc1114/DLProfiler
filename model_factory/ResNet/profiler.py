import torch

from common import Config
from model_factory.ResNet.models import load_resnet50, load_resnet18
from model_factory.hub_common_profile import common_train_template, common_inference_template
from profiler_utils import Profileable, ProfileIterator


def resnet_rand_input(batch_size: int):
    return torch.rand((batch_size, 3, 256, 256), device=Config().local_rank)


def resnet_rand_output():
    return torch.rand((1,), device=Config().local_rank)


class ResNet18Train(Profileable):
    def profile(self, batch_size: int, duration_sec: int) -> ProfileIterator:
        resnet18_model = load_resnet18()
        return common_train_template(resnet18_model, batch_size, duration_sec, resnet_rand_input, resnet_rand_output)


class ResNet18Inference(Profileable):
    def profile(self, batch_size: int, duration_sec: int) -> ProfileIterator:
        resnet18_model = load_resnet18()
        return common_inference_template(resnet18_model, batch_size, duration_sec, resnet_rand_input)


class ResNet50Train(Profileable):
    def profile(self, batch_size: int, duration_sec: int) -> ProfileIterator:
        resnet50_model = load_resnet50()
        return common_train_template(resnet50_model, batch_size, duration_sec, resnet_rand_input, resnet_rand_output)


class ResNet50Inference(Profileable):
    def profile(self, batch_size: int, duration_sec: int) -> ProfileIterator:
        resnet50_model = load_resnet50()
        return common_inference_template(resnet50_model, batch_size, duration_sec, resnet_rand_input)


def do_test():
    from common import process_group
    process_group.setup(0, 1)
    resnet50 = ResNet50Train()
    iterator = resnet50.profile(batch_size=64, duration_sec=10)
    print(iterator.to_dict())
    torch.cuda.empty_cache()
    resnet50 = ResNet50Inference()
    iterator = resnet50.profile(batch_size=64, duration_sec=10)
    print(iterator.to_dict())


if __name__ == '__main__':
    do_test()
