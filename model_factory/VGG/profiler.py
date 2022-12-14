import torch

from common import Config
from model_factory.VGG.models import load_vgg16, load_vgg19
from model_factory.hub_common_profile import common_train_template, common_inference_template, common_checkpoint_template
from profiler_utils import Profileable, ProfileIterator


def vgg_rand_input(batch_size: int):
    return torch.rand((batch_size, 3, 224, 224), device=Config().device)


def vgg_rand_output():
    return torch.rand((1,), device=Config().device)


class VGG16Train(Profileable):
    def profile(self, batch_size: int, duration_sec: int) -> ProfileIterator:
        vgg16_model = load_vgg16()
        return common_train_template(vgg16_model, batch_size, duration_sec, vgg_rand_input, vgg_rand_output)


class VGG16Inference(Profileable):
    def profile(self, batch_size: int, duration_sec: int) -> ProfileIterator:
        vgg16_model = load_vgg16()
        return common_inference_template(vgg16_model, batch_size, duration_sec, vgg_rand_input)


class VGG16Checkpoint(Profileable):
    def profile(self, batch_size: int, duration_sec: int) -> ProfileIterator:
        model = load_vgg16()
        common_train_template(model, batch_size, duration_sec, vgg_rand_input, vgg_rand_output)
        return common_checkpoint_template(model, duration_sec)

class VGG19Train(Profileable):
    def profile(self, batch_size: int, duration_sec: int) -> ProfileIterator:
        vgg16_model = load_vgg19()
        return common_train_template(vgg16_model, batch_size, duration_sec, vgg_rand_input, vgg_rand_output)


class VGG19Inference(Profileable):
    def profile(self, batch_size: int, duration_sec: int) -> ProfileIterator:
        vgg16_model = load_vgg19()
        return common_inference_template(vgg16_model, batch_size, duration_sec, vgg_rand_input)


class VGG19Checkpoint(Profileable):
    def profile(self, batch_size: int, duration_sec: int) -> ProfileIterator:
        model = load_vgg19()
        common_train_template(model, batch_size, duration_sec, vgg_rand_input, vgg_rand_output)
        return common_checkpoint_template(model, duration_sec)

def do_test():
    from common import process_group
    process_group.setup(0, 1)
    profiler = VGG16Train()
    iterator = profiler.profile(batch_size=16, duration_sec=10)
    print(iterator.to_dict())
    torch.cuda.empty_cache()
    profiler = VGG16Inference()
    iterator = profiler.profile(batch_size=16, duration_sec=10)
    print(iterator.to_dict())


if __name__ == '__main__':
    do_test()
